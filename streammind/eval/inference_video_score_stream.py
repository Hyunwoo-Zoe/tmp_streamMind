import os
import re
import math
import json
import csv
import time
import argparse
import warnings
import traceback

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')
# from videollama2 import model_init, x_infer
from videollama2.constants import NUM_FRAMES


import random
# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


import copy
from functools import partial
from typing import Dict, List

import torch

from videollama2.model import Videollama2LlamaForCausalLM, Videollama2MistralForCausalLM, Videollama2MixtralForCausalLM
from videollama2.model.builder import load_pretrained_model
from videollama2.conversation import conv_templates, SeparatorStyle
from videollama2.mm_utils import process_video, tokenizer_MMODAL_token, get_model_name_from_path, KeywordsStoppingCriteria
from videollama2.constants import NUM_FRAMES, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX

def model_init(model_path,model_base = None,model_name="VideoLLaMA2-7B"):
    # model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    # model_name = get_model_name_from_path(model_path) if model_name is None else  model_name
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name)

    if tokenizer.unk_token is not None: 
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES

    if 'vicuna' in model_name.lower():
        # vicuna
        version = 'v1'
    elif 'qwen' in model_name.lower():
        # qwen1.5/qwen2
        version = 'qwen'
    else:
        # mistral/mixtral/llama2
        version = 'llama_2'

    return model, partial(process_video, aspect_ratio=None, processor=processor, num_frames=num_frames), tokenizer, version


def _resolve_model_path(model_path):
    if model_path is None or str(model_path).strip() == "":
        return "DAMO-NLP-SG/VideoLLaMA2-7B"

    model_path = str(model_path).strip()

    # Local path style should exist
    if model_path.startswith("/") or model_path.startswith("./") or model_path.startswith("../"):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Local --model-path not found: {model_path}\n"
                f"Use a valid local checkpoint directory or omit --model-path to use default HF model."
            )
    return model_path


def model_init_cpu(model_path, model_base=None, model_name="VideoLLaMA2-7B"):
    model_path = _resolve_model_path(model_path)
    if model_name is None:
        model_name = get_model_name_from_path(model_path)

    cpu_dtype_name = os.getenv("STREAMMIND_CPU_DTYPE", "bf16").lower()
    cpu_dtype = torch.bfloat16 if cpu_dtype_name == "bf16" else torch.float32

    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, device="cpu", device_map={"": "cpu"}
    )

    # Avoid fp16 kernels on CPU (e.g., LayerNorm half not implemented).
    try:
        model = model.to(device="cpu", dtype=cpu_dtype)
    except Exception:
        model = model.to(device="cpu", dtype=torch.float32)
        cpu_dtype_name = "fp32"

    if os.path.isdir(model_path) and (not os.path.exists(os.path.join(model_path, "mm_projector.bin"))):
        print(
            "[WARN] mm_projector.bin not found in --model-path. "
            "StreamMind gate/mamba weights may be missing, so gate/speech behavior can be unreliable."
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

    return model, partial(process_video, aspect_ratio=None, processor=processor, num_frames=num_frames), tokenizer, version


def model_init_gpu_single(model_path, model_base=None, model_name="VideoLLaMA2-7B", decode_device="cuda:0"):
    model_path = _resolve_model_path(model_path)
    if model_name is None:
        model_name = get_model_name_from_path(model_path)

    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        device=decode_device,
        device_map={"": decode_device},
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

    return model, partial(process_video, aspect_ratio=None, processor=processor, num_frames=num_frames), tokenizer, version


def resolve_model_name_arg(model_path, model_name):
    if model_name is not None and str(model_name).strip() != "":
        return str(model_name).strip()
    try:
        return get_model_name_from_path(_resolve_model_path(model_path))
    except Exception:
        return "VideoLLaMA2-7B"


def infer(model, video, instruct, tokenizer, do_sample=False, version='llama_2',score_video = None, only_cls=False, return_cls=False):

    # 1. vision preprocess (load & transform image or video).
    tensor = video.half().cuda()
    modals = ["video"]

    # 2. text preprocess (tag process & generate prompt).
    modal_token = DEFAULT_MMODAL_TOKEN['VIDEO']
    modal_index = MMODAL_TOKEN_INDEX["VIDEO"]
    instruct = modal_token + '\n' + instruct
    # import pdb
    # pdb.set_trace()
    conv = conv_templates[version].copy()
    conv.append_message(conv.roles[0], instruct)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_index, return_tensors='pt').unsqueeze(0).cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
    # keywords = ["<s>", "</s>"]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    kwargs = {"score_video":score_video,"tokenizer":tokenizer}
    # import pdb
    # pdb.set_trace()
    with torch.inference_mode():
        outputs = model.stream_generate(
            input_ids,
            attention_mask=attention_masks,
            images_or_videos=tensor,
            modal_list=modals,
            do_sample=do_sample,
            temperature=0.2 if do_sample else 0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            only_cls=only_cls,
            return_cls=return_cls,
            **kwargs
        )
        # import pdb
        # pdb.set_trace()
    
    return outputs


def configure_cpu_runtime(num_cores=16, socket_id=0, cores_per_socket=16, interop_threads=1, hide_cuda=True):
    if hide_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_AMX"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.set_num_threads(num_cores)

    # interop threads can only be set once per process before parallel work starts.
    prev_interop = getattr(configure_cpu_runtime, "_interop_set", None)
    req_interop = int(interop_threads)
    if prev_interop is None:
        try:
            torch.set_num_interop_threads(req_interop)
            configure_cpu_runtime._interop_set = req_interop
        except RuntimeError:
            # Keep current runtime value if already initialized elsewhere.
            configure_cpu_runtime._interop_set = int(torch.get_num_interop_threads())
            print(
                f"[WARN] interop threads already initialized; "
                f"using existing value={configure_cpu_runtime._interop_set}"
            )
    elif prev_interop != req_interop:
        print(
            f"[WARN] cannot change interop threads at runtime "
            f"(requested={req_interop}, using={prev_interop})"
        )

    start = socket_id * cores_per_socket
    core_ids = list(range(start, start + num_cores))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(core_ids))
    return core_ids


def maybe_optimize_clip_with_ipex(model, dtype=torch.bfloat16, enable=False):
    if not enable:
        return False
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore
    except Exception:
        return False

    vt = model.get_model().get_vision_tower()
    if not hasattr(vt, "vision_tower"):
        return False
    clip = vt.vision_tower.eval().to("cpu", dtype=dtype)
    for p in clip.parameters():
        p.requires_grad_(False)
    clip = ipex.optimize(clip, dtype=dtype, level="O1", inplace=True)
    vt.vision_tower = clip
    return True


def _resolve_clip_device(device_arg: str) -> torch.device:
    d = torch.device(device_arg)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--clip_device={device_arg} requested but CUDA is not available.")
    return d


def sample_indices(vr, sampling_fps):
    video_fps = float(vr.get_avg_fps())
    stride = max(1, int(round(video_fps / float(sampling_fps))))
    return list(range(0, len(vr), stride))


def build_prompt_cpu(tokenizer, version, instruct):
    modal_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_index = MMODAL_TOKEN_INDEX["VIDEO"]
    conv = conv_templates[version].copy()
    conv.append_message(conv.roles[0], modal_token + "\n" + instruct)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_index, return_tensors="pt").unsqueeze(0)
    attn = input_ids.ne(tokenizer.pad_token_id).long()
    return input_ids, attn


def build_prompt_on_device(tokenizer, version, instruct, device):
    input_ids, attn = build_prompt_cpu(tokenizer, version, instruct)
    return input_ids.to(device), attn.to(device)


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


def setup_stream_cpu_gate_env(args):
    os.environ["STREAMMIND_GATE_MODE"] = "cpu"
    os.environ["STREAMMIND_CPU_DTYPE"] = args.cpu_dtype
    os.environ["STREAMMIND_USE_IPEX"] = os.getenv("STREAMMIND_USE_IPEX", "1")
    os.environ["STREAMMIND_IPEX_COMPILE"] = os.getenv("STREAMMIND_IPEX_COMPILE", "0")
    os.environ["STREAMMIND_MAX_T"] = str(args.max_t)
    os.environ["STREAMMIND_GATE_WINDOW"] = str(args.gate_window)
    if args.disable_debounce:
        os.environ["STREAMMIND_DISABLE_DEBOUNCE"] = "1"
    else:
        os.environ["STREAMMIND_DISABLE_DEBOUNCE"] = "0"
    if args.disable_text_dedup:
        os.environ["STREAMMIND_DISABLE_TEXT_DEDUP"] = "1"
    else:
        os.environ["STREAMMIND_DISABLE_TEXT_DEDUP"] = "0"


def summarize_stream_paced(rows: List[Dict]):
    by = {}
    for r in rows:
        by.setdefault((r["num_cores"], r["input_fps"]), []).append(r)

    summary = []
    for (cores, fps), rs in sorted(by.items(), key=lambda x: (x[0][0], x[0][1])):
        e2e = np.array([r["e2e_ms"] for r in rs], dtype=np.float64)
        queue = np.array([r["queue_ms"] for r in rs], dtype=np.float64)
        proc = np.array([r["proc_ms"] for r in rs], dtype=np.float64)
        clip = np.array([r["clip_ms"] for r in rs], dtype=np.float64)
        offload = np.array([r["offload_ms"] for r in rs], dtype=np.float64)
        gate = np.array([r["gate_compute_ms"] for r in rs], dtype=np.float64)
        mamba = np.array([r["mamba_ms"] for r in rs], dtype=np.float64)
        trans = np.array([r["trans_ms"] for r in rs], dtype=np.float64)
        decode = np.array([r["decode_gpu_ms"] for r in rs], dtype=np.float64)
        wall_inner = np.array([r["wall_inner_ms"] for r in rs], dtype=np.float64)
        miss = np.array([r["miss_deadline"] for r in rs], dtype=np.float64)
        spoke = np.array([r["spoke"] for r in rs], dtype=np.float64)
        cls1 = np.array([r["cls_pred"] for r in rs], dtype=np.float64)

        period_ms = 1000.0 / float(fps)
        p50_proc = float(np.percentile(proc, 50))
        summary.append(
            {
                "num_cores": int(cores),
                "input_fps": int(fps),
                "n": int(len(rs)),
                "avg_e2e_ms": float(e2e.mean()),
                "p50_e2e_ms": float(np.percentile(e2e, 50)),
                "p90_e2e_ms": float(np.percentile(e2e, 90)),
                "p99_e2e_ms": float(np.percentile(e2e, 99)),
                "p50_queue_ms": float(np.percentile(queue, 50)),
                "p50_proc_ms": p50_proc,
                "p50_wall_inner_ms": float(np.percentile(wall_inner, 50)),
                "p50_clip_ms": float(np.percentile(clip, 50)),
                "p50_offload_ms": float(np.percentile(offload, 50)),
                "p50_gate_compute_ms": float(np.percentile(gate, 50)),
                "p50_mamba_ms": float(np.percentile(mamba, 50)),
                "p50_trans_ms": float(np.percentile(trans, 50)),
                "p50_decode_gpu_ms": float(np.percentile(decode, 50)),
                "miss_rate": float(miss.mean()),
                "gate_open_rate": float(cls1.mean()),
                "speak_rate": float(spoke.mean()),
                "utilization_p50": float(p50_proc / period_ms) if period_ms > 0 else None,
                "sustainable_fps_est": float(1000.0 / p50_proc) if p50_proc > 0 else None,
            }
        )
    return summary


def plot_stream_paced(summary, outdir, y_key, ylabel, filename, title):
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    core_list = sorted(set(s["num_cores"] for s in summary))

    plt.figure(figsize=(9, 5.5))
    for cores in core_list:
        sub = sorted([s for s in summary if s["num_cores"] == cores], key=lambda x: x["input_fps"])
        xs = [s["input_fps"] for s in sub]
        ys = [s[y_key] for s in sub]
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=f"{cores} cores")
        for x, y in zip(xs, ys):
            plt.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    plt.xlabel("input_fps (arrival rate)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=220)
    plt.close()


def summarize_latency(rows: List[Dict]):
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


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_latency(summary, outdir, title):
    import matplotlib.pyplot as plt

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


def run_bench_cpu_amx(args):
    core_ids = configure_cpu_runtime(
        num_cores=args.num_cores,
        socket_id=args.socket_id,
        cores_per_socket=args.cores_per_socket,
        interop_threads=args.interop_threads,
    )

    os.environ["STREAMMIND_GATE_MODE"] = "cpu"
    os.environ["STREAMMIND_CPU_DTYPE"] = args.cpu_dtype
    os.environ["STREAMMIND_USE_IPEX"] = os.getenv("STREAMMIND_USE_IPEX", "1")
    os.environ["STREAMMIND_IPEX_COMPILE"] = os.getenv("STREAMMIND_IPEX_COMPILE", "1")
    os.environ["STREAMMIND_MAX_T"] = str(args.max_t)
    os.environ["STREAMMIND_GATE_WINDOW"] = str(args.gate_window)
    os.environ["STREAMMIND_BENCH_GATE_ONLY"] = "1"

    dtype = torch.bfloat16 if args.cpu_dtype == "bf16" else torch.float32

    model, processor, tokenizer, version = model_init_cpu(
        args.model_path, model_base=args.model_base, model_name=args.model_name
    )
    model = model.eval()
    clip_ipex_enabled = maybe_optimize_clip_with_ipex(model, dtype=dtype, enable=args.use_ipex_clip)

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"--video_path not found: {args.video_path}")
    vr = VideoReader(args.video_path, ctx=cpu(0), num_threads=1)
    fps_list = [int(x.strip()) for x in args.fps_list.split(",") if x.strip()]

    instruct = args.instruct or "Describe what is happening right now briefly."
    rows = []
    for fps in fps_list:
        print(f"[BENCH] fps={fps}, cores={args.num_cores}, socket={args.socket_id}, ipex_clip={clip_ipex_enabled}")
        reset_stream_state(model)
        idxs = sample_indices(vr, fps)[: args.warmup + args.steps]
        input_ids, attn = build_prompt_cpu(tokenizer, version, instruct)

        for j, frame_id in enumerate(idxs):
            img = Image.fromarray(vr[frame_id].asnumpy())
            video_tensor = processor([img], num_frames=1).to(device="cpu", dtype=dtype)

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
            if j < args.warmup:
                continue
            rows.append(
                {
                    "sampling_fps": int(fps),
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

    summary = summarize_latency(rows)
    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, "raw.csv")
    sum_path = os.path.join(args.outdir, "summary.csv")
    write_csv(
        raw_path,
        rows,
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
    plot_latency(
        summary,
        args.outdir,
        f"CPU-only StreamMind | cores={args.num_cores} socket={args.socket_id} core_range={core_ids[0]}-{core_ids[-1]} "
        f"dtype={args.cpu_dtype} ipex_clip={int(clip_ipex_enabled)}",
    )
    print(f"[DONE] raw: {raw_path}")
    print(f"[DONE] summary: {sum_path}")
    print(f"[DONE] plot: {os.path.join(args.outdir, 'latency_vs_fps_p50.png')}")


def summarize_clip(rows: List[Dict]):
    by = {}
    for r in rows:
        by.setdefault((r["num_cores"], r["sampling_fps"]), []).append(r)
    summary = []
    for (cores, fps), rs in sorted(by.items(), key=lambda x: (x[0][0], x[0][1])):
        xs = np.array([r["clip_ms"] for r in rs], dtype=np.float64)
        summary.append(
            {
                "num_cores": int(cores),
                "sampling_fps": int(fps),
                "n": int(len(rs)),
                "avg_clip_ms": float(xs.mean()),
                "p50_clip_ms": float(np.percentile(xs, 50)),
                "p90_clip_ms": float(np.percentile(xs, 90)),
                "p99_clip_ms": float(np.percentile(xs, 99)),
                "fps_cap_est": float(1000.0 / np.percentile(xs, 50)) if np.percentile(xs, 50) > 0 else None,
            }
        )
    return summary


def plot_clip_vs_fps(summary, outdir, title):
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    core_list = sorted(set(s["num_cores"] for s in summary))

    plt.figure(figsize=(8.5, 5.5))
    for cores in core_list:
        sub = sorted([s for s in summary if s["num_cores"] == cores], key=lambda x: x["sampling_fps"])
        xs = [s["sampling_fps"] for s in sub]
        ys = [s["p50_clip_ms"] for s in sub]
        plt.plot(xs, ys, marker="o", label=f"{cores} cores")
        for x, y in zip(xs, ys):
            plt.annotate(f"{y:.1f}ms", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    plt.xlabel("sampling_fps")
    plt.ylabel("CLIP latency (ms, p50)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "clip_latency_vs_fps_p50.png"), dpi=220)
    plt.close()


def run_bench_clip_only(args):
    # CPU-only CLIP benchmark: decode/preprocess and clip forward are measured separately.
    os.environ["STREAMMIND_GATE_MODE"] = "cpu"
    os.environ["STREAMMIND_CPU_DTYPE"] = args.cpu_dtype

    dtype = torch.bfloat16 if args.cpu_dtype == "bf16" else torch.float32
    fps_list = [int(x.strip()) for x in args.fps_list.split(",") if x.strip()]
    core_list = [int(x.strip()) for x in args.core_list.split(",") if x.strip()]

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"--video_path not found: {args.video_path}")

    all_rows: List[Dict] = []
    for num_cores in core_list:
        used_cores = configure_cpu_runtime(
            num_cores=num_cores,
            socket_id=args.socket_id,
            cores_per_socket=args.cores_per_socket,
            interop_threads=args.interop_threads,
        )
        print(f"[CLIP-BENCH] cores={num_cores} socket={args.socket_id} core_range={used_cores[0]}-{used_cores[-1]}")
        print("[CLIP-BENCH] loading model on CPU...")

        model, processor, tokenizer, version = model_init_cpu(
            args.model_path, model_base=args.model_base, model_name=args.model_name
        )
        model = model.eval()
        clip_ipex_enabled = maybe_optimize_clip_with_ipex(model, dtype=dtype, enable=args.use_ipex_clip)
        print(f"[CLIP-BENCH] model loaded. ipex_clip={clip_ipex_enabled}")
        vt = model.get_model().get_vision_tower()
        vt = vt.to(device="cpu", dtype=dtype)
        vt.eval()

        vr = VideoReader(args.video_path, ctx=cpu(0), num_threads=1)

        for fps in fps_list:
            print(f"[CLIP-BENCH] cores={num_cores} fps={fps} start")
            idxs = sample_indices(vr, fps)[: args.warmup + args.steps]
            clip_inputs = []
            prep_ms_list = []

            # 1) Decode+preprocess on CPU RAM (not counted in clip_ms)
            for frame_id in idxs:
                t_p0 = time.perf_counter()
                img = Image.fromarray(vr[frame_id].asnumpy())
                x = processor([img], num_frames=1).to(device="cpu", dtype=dtype)
                prep_ms_list.append((time.perf_counter() - t_p0) * 1000.0)
                clip_inputs.append((int(frame_id), x))

            # 2) CLIP forward timing only
            for j, (frame_id, x) in enumerate(clip_inputs):
                with torch.inference_mode():
                    t0 = time.perf_counter()
                    _ = vt(x)
                    clip_ms = (time.perf_counter() - t0) * 1000.0

                if j < args.warmup:
                    continue

                if (j - args.warmup + 1) % max(1, args.steps // 4) == 0:
                    print(
                        f"[CLIP-BENCH] cores={num_cores} fps={fps} "
                        f"step={j - args.warmup + 1}/{args.steps} clip_ms={clip_ms:.2f}"
                    )

                all_rows.append(
                    {
                        "num_cores": int(num_cores),
                        "sampling_fps": int(fps),
                        "frame_id": int(frame_id),
                        "clip_ms": float(clip_ms),
                        "prep_ms": float(prep_ms_list[j]),
                        "ipex_clip": int(clip_ipex_enabled),
                    }
                )

            print(f"[CLIP-BENCH] cores={num_cores} fps={fps} done")

    summary = summarize_clip(all_rows)
    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, "clip_only_raw.csv")
    sum_path = os.path.join(args.outdir, "clip_only_summary.csv")
    write_csv(
        raw_path,
        all_rows,
        ["num_cores", "sampling_fps", "frame_id", "clip_ms", "prep_ms", "ipex_clip"],
    )
    write_csv(
        sum_path,
        summary,
        ["num_cores", "sampling_fps", "n", "avg_clip_ms", "p50_clip_ms", "p90_clip_ms", "p99_clip_ms", "fps_cap_est"],
    )
    plot_clip_vs_fps(
        summary,
        args.outdir,
        f"CLIP CPU-AMX latency vs FPS | cores={args.core_list} socket={args.socket_id} dtype={args.cpu_dtype} ipex={int(args.use_ipex_clip)}",
    )
    print(f"[DONE] clip raw: {raw_path}")
    print(f"[DONE] clip summary: {sum_path}")
    print(f"[DONE] clip plot: {os.path.join(args.outdir, 'clip_latency_vs_fps_p50.png')}")


def summarize_clip_paced(rows: List[Dict]):
    by = {}
    for r in rows:
        clip_device = str(r.get("clip_device", "cpu"))
        by.setdefault((clip_device, r["num_cores"], r["input_fps"]), []).append(r)

    summary = []
    for (clip_device, cores, fps), rs in sorted(by.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        e2e = np.array([r["e2e_ms"] for r in rs], dtype=np.float64)
        queue = np.array([r["queue_ms"] for r in rs], dtype=np.float64)
        proc = np.array([r["proc_ms"] for r in rs], dtype=np.float64)
        clip = np.array([r["clip_ms"] for r in rs], dtype=np.float64)
        prep = np.array([r["prep_ms"] for r in rs], dtype=np.float64)
        miss = np.array([r["miss_deadline"] for r in rs], dtype=np.float64)

        period_ms = 1000.0 / float(fps)
        p50_proc = float(np.percentile(proc, 50))
        interval_s = 1.0 / float(fps)
        # completion offset from first target arrival (step 0)
        completion_offsets = np.array(
            [float(r["step"]) * interval_s + float(r["e2e_ms"]) / 1000.0 for r in rs],
            dtype=np.float64,
        )
        measure_duration_s = float(completion_offsets.max()) if completion_offsets.size > 0 else 0.0
        completed_count = int(len(rs))
        achieved_fps = float(completed_count / measure_duration_s) if measure_duration_s > 0 else 0.0
        summary.append(
            {
                "clip_device": clip_device,
                "num_cores": int(cores),
                "input_fps": int(fps),
                "offered_fps": float(fps),
                "completed_count": completed_count,
                "measure_duration_s": measure_duration_s,
                "achieved_fps": achieved_fps,
                "n": int(len(rs)),
                "avg_e2e_ms": float(e2e.mean()),
                "p50_e2e_ms": float(np.percentile(e2e, 50)),
                "p95_e2e_ms": float(np.percentile(e2e, 95)),
                "p90_e2e_ms": float(np.percentile(e2e, 90)),
                "p99_e2e_ms": float(np.percentile(e2e, 99)),
                "avg_queue_ms": float(queue.mean()),
                "p50_queue_ms": float(np.percentile(queue, 50)),
                "p95_queue_ms": float(np.percentile(queue, 95)),
                "p99_queue_ms": float(np.percentile(queue, 99)),
                "avg_proc_ms": float(proc.mean()),
                "p50_proc_ms": p50_proc,
                "p95_proc_ms": float(np.percentile(proc, 95)),
                "p99_proc_ms": float(np.percentile(proc, 99)),
                "avg_clip_ms": float(clip.mean()),
                "p50_clip_ms": float(np.percentile(clip, 50)),
                "avg_prep_ms": float(prep.mean()),
                "p50_prep_ms": float(np.percentile(prep, 50)),
                "miss_rate": float(miss.mean()),
                "utilization_p50": float(p50_proc / period_ms) if period_ms > 0 else None,
                "sustainable_fps_est": float(1000.0 / p50_proc) if p50_proc > 0 else None,
            }
        )
    return summary


def plot_clip_paced_metric(
    summary,
    outdir,
    y_key,
    ylabel,
    filename,
    title,
    show_identity: bool = False,
):
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    device_list = sorted(set(str(s.get("clip_device", "cpu")) for s in summary))
    core_list = sorted(set(int(s["num_cores"]) for s in summary))

    plt.figure(figsize=(9, 5.5))
    x_all = []
    line_idx = 0
    for clip_device in device_list:
        for cores in core_list:
            sub = sorted(
                [s for s in summary if str(s.get("clip_device", "cpu")) == clip_device and s["num_cores"] == cores],
                key=lambda x: x["input_fps"],
            )
            xs = np.array([s.get("offered_fps", s["input_fps"]) for s in sub], dtype=np.float64)
            ys = np.array([s[y_key] for s in sub], dtype=np.float64)
            if xs.size == 0:
                continue
            order = np.argsort(xs)
            xs = xs[order]
            ys = ys[order]
            x_all.extend(xs.tolist())

            label = f"{cores} cores"
            if len(device_list) > 1:
                label = f"{cores} cores ({clip_device})"
            plt.plot(xs, ys, marker="o", markersize=4.0, linewidth=2.2, label=label)

            # Slightly offset labels so overlapping series remain readable.
            y_offset_points = 8 if (line_idx % 2 == 0) else -10
            for x, y in zip(xs, ys):
                if y_key == "achieved_fps":
                    v = f"{y:.2f}"
                elif y >= 100:
                    v = f"{y:.1f}"
                else:
                    v = f"{y:.2f}"
                plt.annotate(
                    v,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, y_offset_points),
                    ha="center",
                    fontsize=7,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
                )
            line_idx += 1

    if show_identity and len(x_all) > 0:
        x_min = float(min(x_all))
        x_max = float(max(x_all))
        plt.plot(
            [x_min, x_max],
            [x_min, x_max],
            linestyle="--",
            linewidth=1.2,
            color="gray",
            alpha=0.7,
            label="y=x",
        )

    plt.xlabel("offered_fps (arrival rate)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    if len(x_all) > 0:
        x_min = int(min(x_all))
        x_max = int(max(x_all))
        step = 5 if x_max <= 40 else 10
        tick_start = (x_min // step) * step
        if tick_start < x_min:
            tick_start += step
        plt.xticks(np.arange(tick_start, x_max + 1, step))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=220)
    plt.close()


def run_bench_clip_paced(args):
    # Real-time paced benchmark: each step is scheduled by input arrival rate.
    os.environ["STREAMMIND_GATE_MODE"] = "cpu"
    os.environ["STREAMMIND_CPU_DTYPE"] = args.cpu_dtype

    dtype = torch.bfloat16 if args.cpu_dtype == "bf16" else torch.float32
    clip_device = _resolve_clip_device(args.clip_device)
    use_cuda_clip = clip_device.type == "cuda"
    fps_list = [int(x.strip()) for x in args.fps_list.split(",") if x.strip()]
    core_list = [int(x.strip()) for x in args.core_list.split(",") if x.strip()]

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"--video_path not found: {args.video_path}")

    all_rows: List[Dict] = []
    for num_cores in core_list:
        used_cores = configure_cpu_runtime(
            num_cores=num_cores,
            socket_id=args.socket_id,
            cores_per_socket=args.cores_per_socket,
            interop_threads=args.interop_threads,
            hide_cuda=not use_cuda_clip,
        )
        print(
            f"[CLIP-PACED] cores={num_cores} socket={args.socket_id} "
            f"core_range={used_cores[0]}-{used_cores[-1]}"
        )
        print(f"[CLIP-PACED] loading model on CPU (clip_device={clip_device})...")

        model, processor, tokenizer, version = model_init_cpu(
            args.model_path, model_base=args.model_base, model_name=args.model_name
        )
        model = model.eval()
        clip_ipex_enabled = False
        if not use_cuda_clip:
            clip_ipex_enabled = maybe_optimize_clip_with_ipex(model, dtype=dtype, enable=args.use_ipex_clip)
        elif args.use_ipex_clip:
            print("[CLIP-PACED] --use_ipex_clip ignored for CUDA clip device.")
        print(f"[CLIP-PACED] model loaded. ipex_clip={clip_ipex_enabled}")
        vt = model.get_model().get_vision_tower()
        vt = vt.to(device=clip_device, dtype=dtype)
        vt.eval()

        vr = VideoReader(args.video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)

        for fps in fps_list:
            interval_s = 1.0 / float(fps)
            print(f"[CLIP-PACED] cores={num_cores} input_fps={fps} start")

            # Warmup before paced run to reduce one-time compile/init noise.
            warmup_total = max(0, int(args.warmup))
            for j in range(warmup_total):
                frame_id = (args.start_frame + j) % num_frames
                img = Image.fromarray(vr[frame_id].asnumpy())
                x = processor([img], num_frames=1).to(device=clip_device, dtype=dtype, non_blocking=True)
                with torch.inference_mode():
                    _ = vt(x)
                    if use_cuda_clip:
                        torch.cuda.synchronize(clip_device)

            t_stream0 = time.perf_counter()
            for step in range(int(args.steps)):
                target_arrival = t_stream0 + step * interval_s
                now = time.perf_counter()
                if now < target_arrival:
                    time.sleep(target_arrival - now)

                process_start = time.perf_counter()
                queue_ms = max(0.0, (process_start - target_arrival) * 1000.0)

                frame_id = (args.start_frame + warmup_total + step) % num_frames

                t_p0 = time.perf_counter()
                img = Image.fromarray(vr[frame_id].asnumpy())
                x = processor([img], num_frames=1).to(device=clip_device, dtype=dtype, non_blocking=True)
                prep_ms = (time.perf_counter() - t_p0) * 1000.0

                with torch.inference_mode():
                    if use_cuda_clip:
                        torch.cuda.synchronize(clip_device)
                    t_c0 = time.perf_counter()
                    _ = vt(x)
                    if use_cuda_clip:
                        torch.cuda.synchronize(clip_device)
                    clip_ms = (time.perf_counter() - t_c0) * 1000.0

                process_end = time.perf_counter()
                proc_ms = (process_end - process_start) * 1000.0
                e2e_ms = (process_end - target_arrival) * 1000.0
                miss_deadline = int(e2e_ms > (interval_s * 1000.0))

                if (step + 1) % max(1, args.steps // 4) == 0:
                    print(
                        f"[CLIP-PACED] cores={num_cores} fps={fps} step={step + 1}/{args.steps} "
                        f"e2e_ms={e2e_ms:.2f} queue_ms={queue_ms:.2f} clip_ms={clip_ms:.2f}"
                    )

                all_rows.append(
                    {
                        "num_cores": int(num_cores),
                        "input_fps": int(fps),
                        "step": int(step),
                        "frame_id": int(frame_id),
                        "queue_ms": float(queue_ms),
                        "prep_ms": float(prep_ms),
                        "clip_ms": float(clip_ms),
                        "proc_ms": float(proc_ms),
                        "e2e_ms": float(e2e_ms),
                        "miss_deadline": int(miss_deadline),
                        "ipex_clip": int(clip_ipex_enabled),
                        "clip_device": str(clip_device),
                    }
                )

            print(f"[CLIP-PACED] cores={num_cores} input_fps={fps} done")

    summary = summarize_clip_paced(all_rows)
    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, "clip_paced_raw.csv")
    sum_path = os.path.join(args.outdir, "clip_paced_summary.csv")
    write_csv(
        raw_path,
        all_rows,
        [
            "num_cores",
            "input_fps",
            "step",
            "frame_id",
            "queue_ms",
            "prep_ms",
            "clip_ms",
            "proc_ms",
            "e2e_ms",
            "miss_deadline",
            "ipex_clip",
            "clip_device",
        ],
    )
    write_csv(
        sum_path,
        summary,
        [
            "num_cores",
            "clip_device",
            "input_fps",
            "offered_fps",
            "completed_count",
            "measure_duration_s",
            "achieved_fps",
            "n",
            "avg_e2e_ms",
            "p50_e2e_ms",
            "p95_e2e_ms",
            "p90_e2e_ms",
            "p99_e2e_ms",
            "avg_queue_ms",
            "p50_queue_ms",
            "p95_queue_ms",
            "p99_queue_ms",
            "avg_proc_ms",
            "p50_proc_ms",
            "p95_proc_ms",
            "p99_proc_ms",
            "avg_clip_ms",
            "p50_clip_ms",
            "avg_prep_ms",
            "p50_prep_ms",
            "miss_rate",
            "utilization_p50",
            "sustainable_fps_est",
        ],
    )

    plot_clip_paced_metric(
        summary,
        args.outdir,
        y_key="p99_queue_ms",
        ylabel="queueing delay p99 (ms)",
        filename="clip_paced_queue_p99_vs_offered_fps.png",
        title=(
            f"Paced CLIP Queueing (p99) | cores={args.core_list} "
            f"socket={args.socket_id} dtype={args.cpu_dtype} "
            f"ipex={int(args.use_ipex_clip)} clip={args.clip_device}"
        ),
    )
    plot_clip_paced_metric(
        summary,
        args.outdir,
        y_key="p99_e2e_ms",
        ylabel="end-to-end latency p99 (ms)",
        filename="clip_paced_e2e_p99_vs_offered_fps.png",
        title=(
            f"Paced CLIP E2E (p99) | cores={args.core_list} "
            f"socket={args.socket_id} dtype={args.cpu_dtype} "
            f"ipex={int(args.use_ipex_clip)} clip={args.clip_device}"
        ),
    )
    plot_clip_paced_metric(
        summary,
        args.outdir,
        y_key="p50_clip_ms",
        ylabel="CLIP service time p50 (ms)",
        filename="clip_paced_clip_p50_vs_offered_fps.png",
        title=(
            f"Paced CLIP Service Time (p50) | cores={args.core_list} "
            f"socket={args.socket_id} dtype={args.cpu_dtype} "
            f"ipex={int(args.use_ipex_clip)} clip={args.clip_device}"
        ),
    )
    plot_clip_paced_metric(
        summary,
        args.outdir,
        y_key="achieved_fps",
        ylabel="achieved_fps (completion throughput)",
        filename="clip_paced_achieved_fps_vs_offered_fps.png",
        title=(
            f"Paced CLIP Throughput | cores={args.core_list} "
            f"socket={args.socket_id} dtype={args.cpu_dtype} "
            f"ipex={int(args.use_ipex_clip)} clip={args.clip_device}"
        ),
        show_identity=True,
    )
    print(f"[DONE] paced raw: {raw_path}")
    print(f"[DONE] paced summary: {sum_path}")
    print(f"[DONE] paced plot(queue): {os.path.join(args.outdir, 'clip_paced_queue_p99_vs_offered_fps.png')}")
    print(f"[DONE] paced plot(e2e): {os.path.join(args.outdir, 'clip_paced_e2e_p99_vs_offered_fps.png')}")
    print(f"[DONE] paced plot(clip): {os.path.join(args.outdir, 'clip_paced_clip_p50_vs_offered_fps.png')}")
    print(f"[DONE] paced plot(throughput): {os.path.join(args.outdir, 'clip_paced_achieved_fps_vs_offered_fps.png')}")


def run_demo_stream_gate(args):
    # Demo mode: print answer only when gate opens (cls_pred=1).
    dtype = torch.bfloat16 if args.cpu_dtype == "bf16" else torch.float32
    setup_stream_cpu_gate_env(args)
    decode_on_gpu = bool(args.decode_on_gpu)

    used_cores = configure_cpu_runtime(
        num_cores=args.num_cores,
        socket_id=args.socket_id,
        cores_per_socket=args.cores_per_socket,
        interop_threads=args.interop_threads,
        hide_cuda=not decode_on_gpu,
    )
    print(
        f"[DEMO] cores={args.num_cores} socket={args.socket_id} "
        f"core_range={used_cores[0]}-{used_cores[-1]}"
    )

    if decode_on_gpu and not torch.cuda.is_available():
        raise RuntimeError("--decode_on_gpu requested but CUDA is not available.")

    if decode_on_gpu:
        model_name = resolve_model_name_arg(args.model_path, args.model_name)
        print("[DEMO] loading model on GPU (decode), then offloading CLIP+gate path to CPU AMX...")
        model, processor, tokenizer, version = model_init_gpu_single(
            args.model_path,
            model_base=args.model_base,
            model_name=model_name,
            decode_device=args.decode_device,
        )
        decode_device = torch.device(args.decode_device)
    else:
        print("[DEMO] loading model on CPU...")
        model, processor, tokenizer, version = model_init_cpu(
            args.model_path, model_base=args.model_base, model_name=args.model_name
        )
        decode_device = torch.device("cpu")

    model = model.eval()
    clip_ipex_enabled = maybe_optimize_clip_with_ipex(model, dtype=dtype, enable=args.use_ipex_clip)
    vt = model.get_model().get_vision_tower()
    vt = vt.to(device="cpu", dtype=dtype)
    vt.eval()
    print(f"[DEMO] model ready. decode_device={decode_device} ipex_clip={clip_ipex_enabled}")

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"--video_path not found: {args.video_path}")
    vr = VideoReader(args.video_path, ctx=cpu(0), num_threads=1)
    idxs = sample_indices(vr, args.video_fps)[: args.warmup + args.steps]

    reset_stream_state(model)
    input_ids, attn = build_prompt_on_device(tokenizer, version, args.instruct, decode_device)

    rows = []
    for j, frame_id in enumerate(idxs):
        img = Image.fromarray(vr[frame_id].asnumpy())
        t_p0 = time.perf_counter()
        x = processor([img], num_frames=1).to(device="cpu", dtype=dtype)
        prep_ms = (time.perf_counter() - t_p0) * 1000.0

        t0 = time.perf_counter()
        with torch.inference_mode():
            text, cls_pred, metrics = model.stream_generate(
                input_ids,
                attention_mask=attn,
                images_or_videos=x,
                modal_list=["video"],
                do_sample=False,
                temperature=0.0,
                max_new_tokens=args.gen_max_new_tokens,
                use_cache=True,
                return_cls=True,
                return_metrics=True,
                tokenizer=tokenizer,
                score_video=True,
            )
        outer_wall_ms = (time.perf_counter() - t0) * 1000.0

        if j < args.warmup:
            continue

        spoke = int(text is not None and str(text).strip() != "")
        cls_pred_i = int(cls_pred) if cls_pred is not None else 0
        mm = metrics or {}

        if spoke:
            sec = int(frame_id / max(1.0, float(vr.get_avg_fps())))
            print(f"[SPEAK] t={sec//60:02d}:{sec%60:02d} frame={frame_id} cls={cls_pred_i} text={str(text).strip()}")

        rows.append(
            {
                "frame_id": int(frame_id),
                "cls_pred": cls_pred_i,
                "spoke": spoke,
                "prep_ms": float(prep_ms),
                "wall_outer_ms": float(outer_wall_ms),
                "wall_inner_ms": float(mm.get("wall_ms", 0.0) or 0.0),
                "clip_ms": float(mm.get("clip_ms", 0.0) or 0.0),
                "offload_ms": float(mm.get("offload_ms", 0.0) or 0.0),
                "gate_compute_ms": float(mm.get("gate_compute_ms", 0.0) or 0.0),
                "mamba_ms": float(mm.get("mamba_ms", 0.0) or 0.0),
                "trans_ms": float(mm.get("trans_ms", 0.0) or 0.0),
                "decode_gpu_ms": float(mm.get("decode_gpu_ms", 0.0) or 0.0),
                "text": (str(text).strip() if text is not None else ""),
            }
        )

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, "demo_stream_gate_raw.csv")
    write_csv(
        raw_path,
        rows,
        [
            "frame_id",
            "cls_pred",
            "spoke",
            "prep_ms",
            "wall_outer_ms",
            "wall_inner_ms",
            "clip_ms",
            "offload_ms",
            "gate_compute_ms",
            "mamba_ms",
            "trans_ms",
            "decode_gpu_ms",
            "text",
        ],
    )
    n = max(1, len(rows))
    speak_rate = sum(r["spoke"] for r in rows) / n
    gate_rate = sum(r["cls_pred"] for r in rows) / n
    print(f"[DONE] demo raw: {raw_path}")
    print(f"[DONE] demo summary: n={len(rows)} gate_open_rate={gate_rate:.3f} speak_rate={speak_rate:.3f}")


def run_bench_stream_paced(args):
    # Full stream benchmark with real arrival pacing and latency breakdown.
    dtype = torch.bfloat16 if args.cpu_dtype == "bf16" else torch.float32
    setup_stream_cpu_gate_env(args)

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"--video_path not found: {args.video_path}")

    fps_list = [int(x.strip()) for x in args.fps_list.split(",") if x.strip()]
    core_list = [int(x.strip()) for x in args.core_list.split(",") if x.strip()]
    decode_on_gpu = bool(args.decode_on_gpu)
    if decode_on_gpu and not torch.cuda.is_available():
        raise RuntimeError("--decode_on_gpu requested but CUDA is not available.")

    all_rows: List[Dict] = []
    for num_cores in core_list:
        used_cores = configure_cpu_runtime(
            num_cores=num_cores,
            socket_id=args.socket_id,
            cores_per_socket=args.cores_per_socket,
            interop_threads=args.interop_threads,
            hide_cuda=not decode_on_gpu,
        )
        print(
            f"[STREAM-PACED] cores={num_cores} socket={args.socket_id} "
            f"core_range={used_cores[0]}-{used_cores[-1]}"
        )

        if decode_on_gpu:
            model_name = resolve_model_name_arg(args.model_path, args.model_name)
            print("[STREAM-PACED] loading model on GPU (decode), with CLIP+gate on CPU AMX...")
            model, processor, tokenizer, version = model_init_gpu_single(
                args.model_path,
                model_base=args.model_base,
                model_name=model_name,
                decode_device=args.decode_device,
            )
            decode_device = torch.device(args.decode_device)
        else:
            print("[STREAM-PACED] loading model on CPU...")
            model, processor, tokenizer, version = model_init_cpu(
                args.model_path, model_base=args.model_base, model_name=args.model_name
            )
            decode_device = torch.device("cpu")

        model = model.eval()
        clip_ipex_enabled = maybe_optimize_clip_with_ipex(model, dtype=dtype, enable=args.use_ipex_clip)
        vt = model.get_model().get_vision_tower()
        vt = vt.to(device="cpu", dtype=dtype)
        vt.eval()
        print(
            f"[STREAM-PACED] model ready. decode_device={decode_device} "
            f"ipex_clip={clip_ipex_enabled}"
        )

        vr = VideoReader(args.video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)

        for fps in fps_list:
            interval_s = 1.0 / float(fps)
            print(f"[STREAM-PACED] cores={num_cores} input_fps={fps} start")

            reset_stream_state(model)
            input_ids, attn = build_prompt_on_device(tokenizer, version, args.instruct, decode_device)

            warmup_total = max(0, int(args.warmup))
            for j in range(warmup_total):
                frame_id = (args.start_frame + j) % num_frames
                img = Image.fromarray(vr[frame_id].asnumpy())
                x = processor([img], num_frames=1).to(device="cpu", dtype=dtype)
                with torch.inference_mode():
                    _ = model.stream_generate(
                        input_ids,
                        attention_mask=attn,
                        images_or_videos=x,
                        modal_list=["video"],
                        do_sample=False,
                        temperature=0.0,
                        max_new_tokens=args.gen_max_new_tokens,
                        use_cache=True,
                        return_cls=True,
                        return_metrics=False,
                        tokenizer=tokenizer,
                        score_video=True,
                    )

            t_stream0 = time.perf_counter()
            for step in range(int(args.steps)):
                target_arrival = t_stream0 + step * interval_s
                now = time.perf_counter()
                if now < target_arrival:
                    time.sleep(target_arrival - now)

                process_start = time.perf_counter()
                queue_ms = max(0.0, (process_start - target_arrival) * 1000.0)

                frame_id = (args.start_frame + warmup_total + step) % num_frames

                t_p0 = time.perf_counter()
                img = Image.fromarray(vr[frame_id].asnumpy())
                x = processor([img], num_frames=1).to(device="cpu", dtype=dtype)
                prep_ms = (time.perf_counter() - t_p0) * 1000.0

                with torch.inference_mode():
                    text, cls_pred, metrics = model.stream_generate(
                        input_ids,
                        attention_mask=attn,
                        images_or_videos=x,
                        modal_list=["video"],
                        do_sample=False,
                        temperature=0.0,
                        max_new_tokens=args.gen_max_new_tokens,
                        use_cache=True,
                        return_cls=True,
                        return_metrics=True,
                        tokenizer=tokenizer,
                        score_video=True,
                    )

                process_end = time.perf_counter()
                proc_ms = (process_end - process_start) * 1000.0
                e2e_ms = (process_end - target_arrival) * 1000.0
                miss_deadline = int(e2e_ms > (interval_s * 1000.0))

                mm = metrics or {}
                cls_pred_i = int(cls_pred) if cls_pred is not None else 0
                spoke = int(text is not None and str(text).strip() != "")

                if (step + 1) % max(1, args.steps // 4) == 0:
                    print(
                        f"[STREAM-PACED] cores={num_cores} fps={fps} step={step + 1}/{args.steps} "
                        f"e2e={e2e_ms:.1f}ms clip={float(mm.get('clip_ms', 0.0) or 0.0):.1f}ms "
                        f"decode={float(mm.get('decode_gpu_ms', 0.0) or 0.0):.1f}ms cls={cls_pred_i} spoke={spoke}"
                    )

                all_rows.append(
                    {
                        "num_cores": int(num_cores),
                        "input_fps": int(fps),
                        "step": int(step),
                        "frame_id": int(frame_id),
                        "queue_ms": float(queue_ms),
                        "prep_ms": float(prep_ms),
                        "proc_ms": float(proc_ms),
                        "e2e_ms": float(e2e_ms),
                        "miss_deadline": int(miss_deadline),
                        "cls_pred": int(cls_pred_i),
                        "spoke": int(spoke),
                        "wall_inner_ms": float(mm.get("wall_ms", 0.0) or 0.0),
                        "clip_ms": float(mm.get("clip_ms", 0.0) or 0.0),
                        "offload_ms": float(mm.get("offload_ms", 0.0) or 0.0),
                        "xfer_gate_ms": float(mm.get("xfer_gate_ms", 0.0) or 0.0),
                        "gate_compute_ms": float(mm.get("gate_compute_ms", 0.0) or 0.0),
                        "mamba_ms": float(mm.get("mamba_ms", 0.0) or 0.0),
                        "trans_ms": float(mm.get("trans_ms", 0.0) or 0.0),
                        "decode_gpu_ms": float(mm.get("decode_gpu_ms", 0.0) or 0.0),
                        "ipex_clip": int(clip_ipex_enabled),
                        "decode_on_gpu": int(decode_on_gpu),
                    }
                )

            print(f"[STREAM-PACED] cores={num_cores} input_fps={fps} done")

    summary = summarize_stream_paced(all_rows)
    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, "stream_paced_raw.csv")
    sum_path = os.path.join(args.outdir, "stream_paced_summary.csv")
    write_csv(
        raw_path,
        all_rows,
        [
            "num_cores",
            "input_fps",
            "step",
            "frame_id",
            "queue_ms",
            "prep_ms",
            "proc_ms",
            "e2e_ms",
            "miss_deadline",
            "cls_pred",
            "spoke",
            "wall_inner_ms",
            "clip_ms",
            "offload_ms",
            "xfer_gate_ms",
            "gate_compute_ms",
            "mamba_ms",
            "trans_ms",
            "decode_gpu_ms",
            "ipex_clip",
            "decode_on_gpu",
        ],
    )
    write_csv(
        sum_path,
        summary,
        [
            "num_cores",
            "input_fps",
            "n",
            "avg_e2e_ms",
            "p50_e2e_ms",
            "p90_e2e_ms",
            "p99_e2e_ms",
            "p50_queue_ms",
            "p50_proc_ms",
            "p50_wall_inner_ms",
            "p50_clip_ms",
            "p50_offload_ms",
            "p50_gate_compute_ms",
            "p50_mamba_ms",
            "p50_trans_ms",
            "p50_decode_gpu_ms",
            "miss_rate",
            "gate_open_rate",
            "speak_rate",
            "utilization_p50",
            "sustainable_fps_est",
        ],
    )
    plot_stream_paced(
        summary,
        args.outdir,
        y_key="p50_e2e_ms",
        ylabel="end-to-end latency p50 (ms)",
        filename="stream_paced_e2e_vs_input_fps_p50.png",
        title=f"Paced Stream Latency | cores={args.core_list} decode_on_gpu={int(decode_on_gpu)}",
    )
    plot_stream_paced(
        summary,
        args.outdir,
        y_key="p50_clip_ms",
        ylabel="CLIP latency p50 (ms)",
        filename="stream_paced_clip_vs_input_fps_p50.png",
        title=f"Paced Stream CLIP Latency | cores={args.core_list} decode_on_gpu={int(decode_on_gpu)}",
    )
    print(f"[DONE] stream raw: {raw_path}")
    print(f"[DONE] stream summary: {sum_path}")
    print(f"[DONE] stream plot(e2e): {os.path.join(args.outdir, 'stream_paced_e2e_vs_input_fps_p50.png')}")
    print(f"[DONE] stream plot(clip): {os.path.join(args.outdir, 'stream_paced_clip_vs_input_fps_p50.png')}")


def find_video_files(root_path, target_filenames):
    paths = []
    # Traverse the directory structure
    for dirpath, _, filenames in os.walk(root_path):
        # Check if either of the target files is in the current directory
        for target_filename in target_filenames:
            if target_filename in filenames:
                # Append the full path of the found file
                paths.append(os.path.join(dirpath, target_filename))
    return paths

class ScoreDataset(Dataset):

    def __init__(self,):
        print("*****************getting_finetune_score_data******************")
        # target_filenames = ["1_224p.mkv", "2_224p.mkv"]
        target_filenames = ["1_720p.mkv", "2_720p.mkv"]
        # self.score_video_list = find_video_files("/home/v-dingxin/blob/MatchTime/features_video",target_filenames)
        # self.score_video_list = find_video_files("/home/v-dingxin/blob/MatchTime_debug/features_video",target_filenames)
        self.score_video_list = find_video_files("/mnt/input/MatchTime/features_video",target_filenames)
        # import pdb
        # pdb.set_trace()
        self.caption_path_list = []
        self.remove_video_list_id = []
        for video_id, video_path in enumerate(self.score_video_list):
            caption_path = trans_video_2_json(video_path)
            if os.path.exists(caption_path):
                self.caption_path_list.append(caption_path)
            else:
                self.remove_video_list_id.append(video_id) 
        self.score_video_list = [item for idx,item in enumerate(self.score_video_list) if idx not in self.remove_video_list_id]
        # import pdb
        # pdb.set_trace()
        # print(66)
    def __len__(self):
        return len(self.score_video_list)

    def __getitem__(self, idx):
        num_retries = 50
        for _ in range(num_retries):
            try:
                video_path = self.score_video_list[i]
                
            except:
                i = random.randint(0,len(self.score_video_list) -1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # print(video_path, 646446465464654)
        print("*****************score_data_finetune******************")
        # instruct = f'Question: "What is the content of this video?"\nPlease describe the video content in detail based on the provided information.' 
        # instruct = f'Question: "What has happened in the most recent part of the video?"\nPlease describe the video content in detail based on the provided information.' 
        instruct = f'Please describe the video content in detail based on the provided information.' 
        # instruct = f'6666666666666.' 

        return {
            'video_path': video_path,
            'instruct': instruct,
        }


def build_score_eval(args,):
    dataset = ScoreDataset( )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return dataloader


def trans_video_2_json(file_paths):
    # Replace 'features_video' with 'dataset/MatchTime/train'
    new_path = file_paths.replace("features_video", "dataset/MatchTime/train")
    # new_path = file_paths.replace("features_video", "dataset/SN-Caption/test")
    # Replace '1_224p.mkv' or '2_224p.mkv' with 'Labels-caption.json'
    if "1_224p.mkv" in new_path:
        new_path = new_path.replace("1_224p.mkv", "Labels-caption.json")
    elif "2_224p.mkv" in new_path:
        new_path = new_path.replace("2_224p.mkv", "Labels-caption.json")
    elif "1_720p.mkv" in new_path:
        new_path = new_path.replace("1_720p.mkv", "Labels-caption.json")
    elif "2_720p.mkv" in new_path:
        new_path = new_path.replace("2_720p.mkv", "Labels-caption.json")
    
    return new_path

def extract_video_half(video_data_path):
    # Extract the filename from the path
    filename = os.path.basename(video_data_path)
    
    # Use regex to find the number before the underscore
    match = re.match(r"(\d+)_\d+p\.mkv", filename)
    if match:
        return int(match.group(1))
    return None

def calculate_cls_metrics(target_list, predicted_list, tolerance=5):
    # 创建标准范围
    target_ranges = [(t - tolerance, t + tolerance) for t in target_list]

    # 统计 TP 和 FP
    tp = 0
    matched_predicted = set()  # 记录已匹配的预测帧
    for pred in predicted_list:
        for start, end in target_ranges:
            if start <= pred <= end:
                tp += 1
                matched_predicted.add(pred)
                break  # 一个预测帧只匹配一个范围

    fp = len(predicted_list) - len(matched_predicted)

    # 统计 FN
    matched_target = set()
    for t in target_list:
        for pred in predicted_list:
            if (t - tolerance) <= pred <= (t + tolerance):
                matched_target.add(t)
                break  # 一个目标帧只匹配一个预测帧

    fn = len(target_list) - len(matched_target)

    # 计算 Precision, Recall, F1-Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def get_index_stream(start_frame,end_frame , vidoe_fps ,cur_fps = 2):
    
    seg_size = int(vidoe_fps/cur_fps)
    return np.arange(start_frame, end_frame, seg_size, dtype=int)

def read_video_stream(video_path,cur_fps):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    # import pdb
    # pdb.set_trace()
    video_fps = float(vr.get_avg_fps())
    # fps = float(vr.get_avg_fps())
    frame_indices = get_index_stream(start_frame=0,end_frame = max_frame, vidoe_fps = video_fps,cur_fps = cur_fps) 
    return frame_indices, vr


def preprocess_caption_only_caption_data(video_data_path):

    caption_data_path = trans_video_2_json(video_data_path)

    with open(caption_data_path, 'r') as file:
        data = json.load(file)

    timestamp_list = []
    caption_list =  []
    half_list =  []

    half_base = extract_video_half(video_data_path)

    for annotation in data.get('annotations', []):
        gameTime, _ = annotation.get("gameTime",'').split(' - ')
        half = int(gameTime.split(' ')[0])
        if half != half_base:
            continue
        minutes, seconds = map(int, _.split(':'))
        timestamp = minutes * 60 + seconds
        caption_list.append(annotation.get('anonymized', ''))
        timestamp_list.append(timestamp)
        half_list.append(half)
    timestamp_list = timestamp_list[::-1] #让时间从小到大排
    caption_list = caption_list[::-1]
    return timestamp_list, caption_list

def run_inference_time_metric(args):
    model, processor, tokenizer, version = model_init(args.model_path, model_base = args.model_base, model_name = args.model_name)

    val_loader = build_score_eval(args)

    # NOTE: only support batch size 1 for now
    precision_list = []
    recall_list = []
    f1_list = []

    precision_list_10 = []
    recall_list_10 = []
    f1_list_10 = []

    precision_list_1 = []
    recall_list_1 = []
    f1_list_1 = []

    for i, line in enumerate(tqdm(val_loader)):
        # if i > 1:
        #     continue
        video_path = line['video_path'][0]
        instruct = line['instruct'][0]
        # if i < 5:                                                          
        #     continue
        # video_path = "/home/v-dingxin/blob/MatchTime/features_video/england_epl_2014-2015/2015-02-21_-_18-00_Chelsea_1_-_1_Burnley/1_224p.mkv"
        timestamp_list, caption_list = preprocess_caption_only_caption_data(video_path)
        pred_timestamp_list = []
        video_frame ,vr = read_video_stream(video_path,2)
        prev_cls_pred = 0
        last_text = None

        for frame_id in video_frame:
            img = Image.fromarray(vr[frame_id].asnumpy())
            images_group = [img]
            video_frame = processor(images_group, num_frames=len(images_group))

            pred, cls_pred = infer(
                video=video_frame,
                instruct=instruct,
                model=model,
                tokenizer=tokenizer,
                do_sample=False,
                version=version,
                score_video=True,
                only_cls=False,     # 항상 False
                return_cls=True,    # cls_pred 같이 받기
            )

            if cls_pred == 1 and prev_cls_pred == 0:
                if pred is not None and pred != "" and pred != last_text:
                    print("The content of the video until {}:{}  is: {}:".format(
                        frame_id//25//60, frame_id//25%60, pred
                    ))
                    pred_timestamp_list.append(frame_id//25)
                    last_text = pred

            prev_cls_pred = cls_pred


        model.frame_feature = None
        model.past_review_caption = None
        model.interval_id_list = []
        # import pdb
        # pdb.set_trace()
        precision, recall, f1 = calculate_cls_metrics(timestamp_list, pred_timestamp_list)
        precision_10, recall_10, f1_10 = calculate_cls_metrics(timestamp_list, pred_timestamp_list,10)
        precision_1, recall_1, f1_1 = calculate_cls_metrics(timestamp_list, pred_timestamp_list,1)

        precision_list.append(precision)
        precision_list_10.append(precision_10)
        precision_list_1.append(precision_1)

        recall_list.append(recall)
        recall_list_10.append(recall_10)
        recall_list_1.append(recall_1)

        f1_list.append(f1)
        f1_list_10.append(f1_10)
        f1_list_1.append(f1_1)

        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
        print(f"Precision_10: {precision_10:.2f}, Recall_10: {recall_10:.2f}, F1-Score_10: {f1_10:.2f}")
        print(f"Precision_1: {precision_1:.2f}, Recall_10: {recall_1:.2f}, F1-Score_10: {f1_1:.2f}")

    print(f"final Precision: {sum(precision_list)/len(precision_list):.2f}, Recall: {sum(recall_list)/len(recall_list):.2f}, F1-Score: {sum(f1_list)/len(f1_list):.2f}")
    print(f"final Precision_10: {sum(precision_list_10)/len(precision_list_10):.2f}, Recall_10: {sum(recall_list_10)/len(recall_list_10):.2f}, F1-Score_10: {sum(f1_list_10)/len(f1_list_10):.2f}")
    print(f"final Precision_10: {sum(precision_list_1)/len(precision_list_1):.2f}, Recall_10: {sum(recall_list_1)/len(recall_list_1):.2f}, F1-Score_10: {sum(f1_list_1)/len(f1_list_1):.2f}")
    #precision:衡量模型预测的帧中有多少是正确的。,recall:衡量模型是否覆盖了大部分标准帧。
    #TP：模型预测的帧落在任意标准帧范围内的数量。FP：模型预测的帧未落在任何标准帧范围内的数量。FN:标准帧范围内没有被模型预测到的帧数量。
    # ans_file.close()

def is_dataset_caption(timestamp_id,target_timestamp_list,tolerance=5):
    target_ranges = [(t - tolerance, t + tolerance) for t in target_timestamp_list]

    # 统计 TP 和 FP
    tp = 0
    matched_predicted = set()  # 记录已匹配的预测帧
    for i, (start, end) in enumerate(target_ranges):
        if start <= timestamp_id <= end:
            return True,i
    return False, None


def run_inference_caption_metric(args):
    from score_single import calculate_metrics
    model, processor, tokenizer, version = model_init(args.model_path, model_base = args.model_base, model_name = args.model_name)

    val_loader = build_score_eval(args)

    # NOTE: only support batch size 1 for now
    pred_caption_list = {}
    target_caption_list = {}
    caption_id = 0
    for i, line in enumerate(tqdm(val_loader)):
        # continue
        video_path = line['video_path'][0]
        instruct = line['instruct'][0]
        # if i > 1:                                                          
        #     continue
        # video_path = "/home/v-dingxin/blob/MatchTime_debug/features_video/england_epl_2014-2015/2015-02-21_-_18-00_Crystal_Palace_1_-_2_Arsenal/2_720p.mkv"
        timestamp_list, caption_list = preprocess_caption_only_caption_data(video_path)
        pred_timestamp_list = []
        frame_interval = 1 / args.video_fps

        total_video_frame ,vr = read_video_stream(video_path,args.video_fps)
        cur_min = -1
        cur_sec = -1
        for frame_id in total_video_frame:
            if frame_id < 200:
                continue
            if ((frame_id//25//60 == cur_min) and (frame_id//25%60 > cur_sec)) or (frame_id//25//60 > cur_min) : 
                img = Image.fromarray(vr[frame_id].asnumpy())
                images_group = [img]

                # frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # cv2.imshow("video",frame)

                # images_group = [expand2square(img, tuple(int(x*255) for x in self.processor.image_mean)) for img in images_group]
                video_frame = processor(images_group,num_frames=len(images_group))
                # import pdb
                # pdb.set_trace()
                # video_tensor = process_score_video(1112,processor,"/home/v-dingxin/blob/MatchTime_debug/features_video/england_epl_2014-2015/2015-04-11_-_19-30_Burnley_0_-_1_Arsenal/2_224p.mkv",2 )
                # video_tensor  = torch.randn(84,3,336,336)
                pred = infer(
                    video=video_frame,
                    instruct=instruct,
                    model=model,
                    tokenizer=tokenizer,
                    do_sample=False,
                    version=version,
                    score_video=True
                )
                # import pdb
                # pdb.set_trace()
                # print(pred)
                if pred is not None and pred != "":
                    cur_min = frame_id // 25 // 60
                    cur_sec = frame_id // 25 % 60
                    # print("The content of the video until {}:{}  is: {}:".format(frame_id//25//60,frame_id//25%60,pred))
                    isdataset, caption_id =  is_dataset_caption(frame_id//25,timestamp_list,tolerance=5)
                    if isdataset:
                        print("The content of the video until {}:{}  is: {}:".format(frame_id//25//60,frame_id//25%60,pred))
                        # pred_timestamp_list.append(frame_id//25)
                        pred_caption_list[caption_id] = [pred]
                        target_caption_list[caption_id] = [caption_list[caption_id]]
                # if cv2.waitKey(int(frame_interval * 1000)) & 0xFF == ord('q'):
                #     break
        # cv2.destroyAllWindows()
        model.frame_feature = None
        model.past_review_caption = None

    result = calculate_metrics(pred_caption_list,target_caption_list)
    print(result)
    #precision:衡量模型预测的帧中有多少是正确的。,recall:衡量模型是否覆盖了大部分标准帧。
    #TP：模型预测的帧落在任意标准帧范围内的数量。FP：模型预测的帧未落在任何标准帧范围内的数量。FN:标准帧范围内没有被模型预测到的帧数量。
    # ans_file.close()


def run_inference_timediff_fluency_ppl_metric(args):
    # 计算videollm-online的指标

    # timediff（eos的出错数量）: 计算整个video的eos数量- 该位置误识别的
    # turn_stream_masked_pred_mask = turn_stream_masked_score.argmax(dim=-1) != frame_token_interval_id #
    #frame_diff = turn_stream_mask.sum() - turn_stream_masked_pred_mask.nonzero()[0,0] - 1

    #llm_ppl = ppl 

    #fluency = correct_eos_token_num + correct_caption_token_num
    model, processor, tokenizer, version = model_init(args.model_path, model_base = args.model_base, model_name = args.model_name)

    val_loader = build_score_eval(args)

    # NOTE: only support batch size 1 for now
    pred_caption_list = {}
    target_caption_list = {}
    caption_id = 0
    for i, line in enumerate(tqdm(val_loader)):
        # continue
        video_path = line['video_path'][0]
        instruct = line['instruct'][0]
        # if i > 1:                                                          
        #     continue
        # video_path = "/home/v-dingxin/blob/MatchTime_debug/features_video/england_epl_2014-2015/2015-02-21_-_18-00_Crystal_Palace_1_-_2_Arsenal/2_720p.mkv"
        timestamp_list, caption_list = preprocess_caption_only_caption_data(video_path)
        pred_timestamp_list = []
        frame_interval = 1 / args.video_fps

        total_video_frame ,vr = read_video_stream(video_path,args.video_fps)
        cur_min = -1
        cur_sec = -1
        for frame_id in total_video_frame:
            if frame_id < 200:
                continue
            if ((frame_id//25//60 == cur_min) and (frame_id//25%60 > cur_sec)) or (frame_id//25//60 > cur_min) : 
                img = Image.fromarray(vr[frame_id].asnumpy())
                images_group = [img]

                # frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # cv2.imshow("video",frame)

                # images_group = [expand2square(img, tuple(int(x*255) for x in self.processor.image_mean)) for img in images_group]
                video_frame = processor(images_group,num_frames=len(images_group))
                # import pdb
                # pdb.set_trace()
                # video_tensor = process_score_video(1112,processor,"/home/v-dingxin/blob/MatchTime_debug/features_video/england_epl_2014-2015/2015-04-11_-_19-30_Burnley_0_-_1_Arsenal/2_224p.mkv",2 )
                # video_tensor  = torch.randn(84,3,336,336)
                pred = infer(
                    video=video_frame,
                    instruct=instruct,
                    model=model,
                    tokenizer=tokenizer,
                    do_sample=False,
                    version=version,
                    score_video=True
                )
                


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--model-path', help='', required=True)
    # parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    # parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    # parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    # parser.add_argument("--num-chunks", type=int, default=1)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--device", type=str, required=False, default='cuda:0')
    # parser.add_argument("--batch-size", type=int, default=1)
    # parser.add_argument("--num-workers", type=int, default=8)
    # args = parser.parse_args()

    # run_inference(args)
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', default=None)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--model-base', default=None)
    parser.add_argument('--eval-cls', default=None)
    parser.add_argument('--eval-caption', default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--video_fps", type=int, default=2)

    # CPU AMX benchmark options (integrated)
    parser.add_argument("--bench_cpu_amx", action="store_true")
    parser.add_argument("--bench_clip_only", action="store_true")
    parser.add_argument("--bench_clip_paced", action="store_true")
    parser.add_argument("--bench_stream_paced", action="store_true")
    parser.add_argument("--demo_stream_gate", action="store_true")
    parser.add_argument("--video_path", type=str, default="assets/blind_com_demo.mp4")
    parser.add_argument("--fps_list", type=str, default="1,2,4,8,16")
    parser.add_argument("--core_list", type=str, default="16,8")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--outdir", type=str, default="result/cpu_clip_amx_16core")
    parser.add_argument("--num_cores", type=int, default=16)
    parser.add_argument("--socket_id", type=int, default=0)
    parser.add_argument("--cores_per_socket", type=int, default=16)
    parser.add_argument("--interop_threads", type=int, default=1)
    parser.add_argument("--cpu_dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--clip_device", type=str, default="cpu")
    parser.add_argument("--use_ipex_clip", action="store_true")
    parser.add_argument("--decode_on_gpu", action="store_true")
    parser.add_argument("--decode_device", type=str, default="cuda:0")
    parser.add_argument("--gen_max_new_tokens", type=int, default=64)
    parser.add_argument("--disable_debounce", action="store_true")
    parser.add_argument("--disable_text_dedup", action="store_true")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_t", type=int, default=64)
    parser.add_argument("--gate_window", type=int, default=64)
    parser.add_argument("--instruct", type=str, default="Describe what is happening right now briefly.")

    args = parser.parse_args()
    if args.demo_stream_gate:
        run_demo_stream_gate(args)
    elif args.bench_stream_paced:
        run_bench_stream_paced(args)
    elif args.bench_clip_paced:
        run_bench_clip_paced(args)
    elif args.bench_clip_only:
        run_bench_clip_only(args)
    elif args.bench_cpu_amx:
        run_bench_cpu_amx(args)
    elif args.eval_cls:
        run_inference_time_metric(args)
    elif args.eval_caption:
        run_inference_caption_metric(args)
