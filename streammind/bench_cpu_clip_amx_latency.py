import argparse
import csv
import os
import time
from functools import partial
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu

from streammind.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from streammind.conversation import conv_templates
from streammind.mm_utils import tokenizer_MMODAL_token, process_video, get_model_name_from_path
from streammind.model.builder import load_pretrained_model
from streammind.constants import NUM_FRAMES


def configure_cpu_runtime(num_cores: int, socket_id: int, cores_per_socket: int, interop_threads: int) -> List[int]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_AMX"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(interop_threads)

    start = socket_id * cores_per_socket
    cores = list(range(start, start + num_cores))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cores))
    return cores


def maybe_optimize_clip_with_ipex(model: torch.nn.Module, dtype: torch.dtype, use_ipex: bool) -> bool:
    if not use_ipex:
        return False
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore
    except Exception:
        return False

    vt = model.get_model().get_vision_tower()
    clip = vt.vision_tower.eval().to("cpu", dtype=dtype)
    for p in clip.parameters():
        p.requires_grad_(False)
    clip = ipex.optimize(clip, dtype=dtype, level="O1", inplace=True)
    vt.vision_tower = clip
    return True


def model_init_cpu(model_path=None, model_name="VideoLLaMA2-7B"):
    model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path) if model_name is None else model_name
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device="cpu",
        device_map={"": "cpu"},
    )

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    if "vicuna" in model_name.lower():
        version = "v1"
    elif "qwen" in model_name.lower():
        version = "qwen"
    else:
        version = "llama_2"

    processor_fn = partial(process_video, aspect_ratio=None, processor=processor, num_frames=num_frames)
    return model, processor_fn, tokenizer, version


def sample_indices(vr: VideoReader, sampling_fps: int) -> List[int]:
    video_fps = float(vr.get_avg_fps())
    stride = max(1, int(round(video_fps / float(sampling_fps))))
    return list(range(0, len(vr), stride))


def build_prompt_once(tokenizer, version: str, instruct: str):
    modal_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_index = MMODAL_TOKEN_INDEX["VIDEO"]
    conv = conv_templates[version].copy()
    conv.append_message(conv.roles[0], modal_token + "\n" + instruct)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_MMODAL_token(
        prompt, tokenizer, modal_index, return_tensors="pt"
    ).unsqueeze(0)
    attn = input_ids.ne(tokenizer.pad_token_id).long()
    return input_ids, attn


def reset_stream_state(model):
    if hasattr(model, "frame_feature"):
        model.frame_feature = None
    if hasattr(model, "past_review_caption"):
        model.past_review_caption = None
    if hasattr(model, "interval_id_list"):
        model.interval_id_list = []
    if hasattr(model, "_gate_prev"):
        model._gate_prev = None
    if hasattr(model, "_last_spoken_text"):
        model._last_spoken_text = ""


def run_one_fps(
    model,
    processor,
    tokenizer,
    version: str,
    vr: VideoReader,
    sampling_fps: int,
    warmup: int,
    steps: int,
    instruct: str,
    dtype: torch.dtype,
) -> List[Dict]:
    reset_stream_state(model)
    rows: List[Dict] = []

    idxs = sample_indices(vr, sampling_fps)[: warmup + steps]
    input_ids, attn = build_prompt_once(tokenizer, version, instruct)

    for j, frame_id in enumerate(idxs):
        img = Image.fromarray(vr[frame_id].asnumpy())
        video_tensor = processor([img], num_frames=1).to(dtype=dtype, device="cpu")

        t0 = time.perf_counter()
        with torch.inference_mode():
            _, cls_pred, metrics = model.stream_generate(
                input_ids,
                attention_mask=attn,
                images_or_videos=video_tensor,
                modal_list=["video"],
                do_sample=False,
                temperature=0.0,
                max_new_tokens=1,
                use_cache=False,
                only_cls=True,
                return_cls=True,
                return_metrics=True,
                bench_gate_only=True,
                tokenizer=tokenizer,
                score_video=True,
            )
        wall_ms = (time.perf_counter() - t0) * 1000.0

        if j < warmup:
            continue

        rows.append(
            {
                "sampling_fps": int(sampling_fps),
                "frame_id": int(frame_id),
                "cls_pred": int(cls_pred) if cls_pred is not None else -1,
                "wall_ms": float(wall_ms),
                "clip_ms": float((metrics or {}).get("clip_ms") or 0.0),
                "offload_ms": float((metrics or {}).get("offload_ms") or 0.0),
                "xfer_gate_ms": float((metrics or {}).get("xfer_gate_ms") or 0.0),
                "gate_compute_ms": float((metrics or {}).get("gate_compute_ms") or 0.0),
                "mamba_ms": float((metrics or {}).get("mamba_ms") or 0.0),
                "trans_ms": float((metrics or {}).get("trans_ms") or 0.0),
            }
        )
    return rows


def summarize(rows: List[Dict]) -> List[Dict]:
    by = {}
    for r in rows:
        by.setdefault(r["sampling_fps"], []).append(r)

    summary = []
    for fps in sorted(by.keys()):
        rs = by[fps]
        xs = np.array([r["wall_ms"] for r in rs], dtype=np.float64)
        summary.append(
            {
                "sampling_fps": int(fps),
                "n": int(len(rs)),
                "avg_wall_ms": float(xs.mean()),
                "p50_wall_ms": float(np.percentile(xs, 50)),
                "p90_wall_ms": float(np.percentile(xs, 90)),
                "p99_wall_ms": float(np.percentile(xs, 99)),
            }
        )
    return summary


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_latency(summary: List[Dict], outdir: str, title: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    xs = [s["sampling_fps"] for s in summary]
    ys = [s["p50_wall_ms"] for s in summary]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}ms", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    plt.xlabel("sampling_fps")
    plt.ylabel("latency (ms, p50)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "latency_vs_fps_p50.png"), dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="assets/blind_com_demo.mp4")
    ap.add_argument("--fps_list", default="1,2,4,8,16")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--outdir", default="result/cpu_clip_amx_16core")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--model_name", default="VideoLLaMA2-7B")
    ap.add_argument("--instruct", default="Describe what is happening right now briefly.")

    # CPU/socket settings
    ap.add_argument("--num_cores", type=int, default=16)
    ap.add_argument("--socket_id", type=int, default=0)
    ap.add_argument("--cores_per_socket", type=int, default=16)
    ap.add_argument("--interop_threads", type=int, default=1)

    # AMX/precision
    ap.add_argument("--cpu_dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--use_ipex_clip", action="store_true")
    args = ap.parse_args()

    used_cores = configure_cpu_runtime(
        num_cores=args.num_cores,
        socket_id=args.socket_id,
        cores_per_socket=args.cores_per_socket,
        interop_threads=args.interop_threads,
    )

    os.environ["STREAMMIND_GATE_MODE"] = "cpu"
    os.environ["STREAMMIND_USE_IPEX"] = "1"
    os.environ["STREAMMIND_IPEX_COMPILE"] = "0"
    os.environ["STREAMMIND_CPU_DTYPE"] = args.cpu_dtype
    os.environ["STREAMMIND_MAX_T"] = "64"
    os.environ["STREAMMIND_GATE_WINDOW"] = "64"
    os.environ["STREAMMIND_BENCH_GATE_ONLY"] = "1"

    dtype = torch.bfloat16 if args.cpu_dtype == "bf16" else torch.float32

    model, processor, tokenizer, version = model_init_cpu(model_path=args.model_path, model_name=args.model_name)
    model = model.eval()

    clip_ipex_enabled = maybe_optimize_clip_with_ipex(model, dtype=dtype, use_ipex=args.use_ipex_clip)

    vr = VideoReader(args.video, ctx=cpu(0), num_threads=1)
    fps_list = [int(x.strip()) for x in args.fps_list.split(",") if x.strip()]

    all_rows: List[Dict] = []
    for fps in fps_list:
        print(f"[RUN] fps={fps}, cores={args.num_cores}, socket={args.socket_id}, ipex_clip={clip_ipex_enabled}")
        rows = run_one_fps(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            version=version,
            vr=vr,
            sampling_fps=fps,
            warmup=args.warmup,
            steps=args.steps,
            instruct=args.instruct,
            dtype=dtype,
        )
        all_rows.extend(rows)

    summary = summarize(all_rows)

    raw_path = os.path.join(args.outdir, "raw.csv")
    sum_path = os.path.join(args.outdir, "summary.csv")
    write_csv(
        raw_path,
        all_rows,
        [
            "sampling_fps",
            "frame_id",
            "cls_pred",
            "wall_ms",
            "clip_ms",
            "offload_ms",
            "xfer_gate_ms",
            "gate_compute_ms",
            "mamba_ms",
            "trans_ms",
        ],
    )
    write_csv(sum_path, summary, ["sampling_fps", "n", "avg_wall_ms", "p50_wall_ms", "p90_wall_ms", "p99_wall_ms"])

    title = (
        f"CPU-only StreamMind Latency vs FPS | cores={args.num_cores} "
        f"(socket {args.socket_id}, used={used_cores[0]}-{used_cores[-1]}), dtype={args.cpu_dtype}, "
        f"clip_ipex={int(clip_ipex_enabled)}"
    )
    plot_latency(summary, args.outdir, title)

    print(f"[DONE] raw: {raw_path}")
    print(f"[DONE] summary: {sum_path}")
    print(f"[DONE] plot: {os.path.join(args.outdir, 'latency_vs_fps_p50.png')}")


if __name__ == "__main__":
    main()
