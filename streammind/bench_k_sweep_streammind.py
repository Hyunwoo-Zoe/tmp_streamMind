# streammind/bench_gate_offload_latency.py
# Gate offload/latency microbench (StreamMind / VideoLLaMA2)

import os, csv, time, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from decord import VideoReader, cpu

from streammind import model_init
from streammind.conversation import conv_templates
from streammind.mm_utils import tokenizer_MMODAL_token
from streammind.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX


def sample_indices(vr, sampling_fps: int):
    video_fps = float(vr.get_avg_fps())
    stride = max(1, int(round(video_fps / float(sampling_fps))))
    return list(range(0, len(vr), stride))


def build_prompt_once(tokenizer, version, instruct: str):
    modal_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_index = MMODAL_TOKEN_INDEX["VIDEO"]
    conv = conv_templates[version].copy()
    conv.append_message(conv.roles[0], modal_token + "\n" + instruct)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_MMODAL_token(
        prompt, tokenizer, modal_index, return_tensors="pt"
    ).unsqueeze(0).cuda()
    attn = input_ids.ne(tokenizer.pad_token_id).long().cuda()
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


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _get_stats(rs, key):
    # 키가 없거나 None인 경우를 안전하게 처리
    xs = [float(r[key]) for r in rs if r.get(key) is not None]
    if not xs:
        return None, None, None
    
    avg = float(np.mean(xs))
    p50 = float(np.median(xs))
    p99 = float(np.percentile(xs, 99)) 
    return avg, p50, p99

def summarize(rows):
    by = {}
    for r in rows:
        k = (r["gate_mode"], r["sampling_fps"])
        by.setdefault(k, []).append(r)

    summary = []
    for (mode, fps), rs in sorted(by.items(), key=lambda x: (x[0][0], x[0][1])):
        wall_avg, wall_p50, wall_p99 = _get_stats(rs, "wall_ms")
        
        _, clip_p50, _ = _get_stats(rs, "clip_ms")
        _, offload_p50, _ = _get_stats(rs, "offload_ms")
        _, xfer_p50, _ = _get_stats(rs, "xfer_gate_ms")
        _, comp_p50, _ = _get_stats(rs, "gate_comp_ms")
        
        # [수정] 세부 내역 집계 추가
        _, mamba_p50, _ = _get_stats(rs, "mamba_ms")
        _, trans_p50, _ = _get_stats(rs, "trans_ms")

        summary.append({
            "gate_mode": mode,
            "sampling_fps": int(fps),
            "n": len(rs),
            
            "avg_wall_ms": wall_avg,  
            "p50_wall_ms": wall_p50,   
            "p99_wall_ms": wall_p99, 

            "p50_clip_ms": clip_p50 if clip_p50 is not None else 0.0,
            "p50_offload_ms": offload_p50 if offload_p50 is not None else 0.0,
            "p50_xfer_gate_ms": xfer_p50 if xfer_p50 is not None else 0.0,
            "p50_gate_comp_ms": comp_p50 if comp_p50 is not None else 0.0,
            
            # [수정] 세부 내역 추가
            "p50_mamba_ms": mamba_p50 if mamba_p50 is not None else 0.0,
            "p50_trans_ms": trans_p50 if trans_p50 is not None else 0.0,
            
            "sustainable_fps_est": (1000.0 / wall_p50) if (wall_p50 and wall_p50 > 0) else None,
        })
    return summary


def _annot(xs, ys, fmt="{:.0f}ms", dy=10):
    for x, y in zip(xs, ys):
        if y is None:
            continue
        plt.annotate(fmt.format(y), (x, y), textcoords="offset points",
                     xytext=(0, dy), ha="center", fontsize=9)

def plot_summary(outdir, summary, title_suffix: str, logy: bool):
    os.makedirs(outdir, exist_ok=True)
    modes = sorted(set(s["gate_mode"] for s in summary))

    # 1) p50 wall latency (Line Plot)
    plt.figure(figsize=(9, 6))
    for mode in modes:
        xs = [s["sampling_fps"] for s in summary if s["gate_mode"] == mode]
        ys = [s["p50_wall_ms"] for s in summary if s["gate_mode"] == mode]
        plt.plot(xs, ys, marker="o", label=mode)
        _annot(xs, ys, fmt="{:.0f}ms")
    plt.xlabel("sampling_fps")
    plt.ylabel("p50 wall latency (ms)")
    plt.title(f"Gate latency p50 | {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(outdir, "p50_wall_ms.png"), dpi=200)
    plt.close()

    # 2) Component Breakdown (Stacked Bar Chart)
    for mode in modes:
        subset = sorted([s for s in summary if s["gate_mode"] == mode], key=lambda x: x["sampling_fps"])
        if not subset: continue

        x_labels = [str(s["sampling_fps"]) for s in subset]
        clip = np.array([s.get("p50_clip_ms", 0.0) for s in subset])
        
        # Transfer 계열 합치기
        transfer = np.array([s.get("p50_offload_ms", 0.0) for s in subset]) + \
                   np.array([s.get("p50_xfer_gate_ms", 0.0) for s in subset])
        
        # [수정] Mamba vs Trans 분리
        mamba = np.array([s.get("p50_mamba_ms", 0.0) for s in subset])
        trans = np.array([s.get("p50_trans_ms", 0.0) for s in subset])
        
        # Fallback: 만약 mamba/trans 데이터가 아예 없으면 gate_comp를 사용 (하지만 이제 데이터가 있을 것임)
        fallback_gate = np.array([s.get("p50_gate_comp_ms", 0.0) for s in subset])
        if np.sum(mamba) == 0 and np.sum(trans) == 0:
            mamba = fallback_gate 

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 순서: CLIP -> Transfer -> Mamba -> Transformer
        p1 = ax.bar(x_labels, clip, label="CLIP (Vision)", color="#1f77b4", alpha=0.85)
        p2 = ax.bar(x_labels, transfer, bottom=clip, label="Transfer (D2H)", color="#ff7f0e", alpha=0.85)
        p3 = ax.bar(x_labels, mamba, bottom=clip+transfer, label="Mamba SSM", color="#2ca02c", alpha=0.85)
        p4 = ax.bar(x_labels, trans, bottom=clip+transfer+mamba, label="Transformer (Gate)", color="#d62728", alpha=0.85)
        
        ax.set_ylabel("Latency (ms)")
        ax.set_xlabel("Sampling FPS")
        ax.set_title(f"Detailed Breakdown - {mode.upper()} | {title_suffix}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # 총 시간 표시
        total_heights = clip + transfer + mamba + trans
        for i, val in enumerate(total_heights):
            ax.text(i, val + (val * 0.01), f"{val:.1f}ms", ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        filename = f"stacked_breakdown_detail_{mode}.png"
        plt.savefig(os.path.join(outdir, filename), dpi=200)
        print(f"[PLOT] Saved breakdown: {os.path.join(outdir, filename)}")
        plt.close()


def run_one_setting(model, processor, tokenizer, version, vr, sampling_fps, steps, warmup, realtime, instruct, gate_mode, rows):
    reset_stream_state(model)

    idxs = sample_indices(vr, sampling_fps)
    idxs = idxs[: (warmup + steps)]
    input_ids, attn = build_prompt_once(tokenizer, version, instruct=instruct)

    t_next = time.perf_counter()

    for j, frame_id in enumerate(idxs):
        if realtime and sampling_fps > 0:
            t_next += 1.0 / float(sampling_fps)
            now = time.perf_counter()
            if t_next > now:
                time.sleep(t_next - now)

        img = Image.fromarray(vr[frame_id].asnumpy())
        video_tensor = processor([img], num_frames=1)
        video_tensor = video_tensor.half().cuda()

        with torch.inference_mode():
            t0 = time.perf_counter()
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
            t1 = time.perf_counter()

        if j < warmup:
            continue

        # [수정] metrics에서 mamba_ms와 trans_ms를 추출해서 row에 추가합니다.
        rows.append({
            "gate_mode": gate_mode,
            "sampling_fps": int(sampling_fps),
            "frame_id": int(frame_id),
            "cls_pred": int(cls_pred) if cls_pred is not None else -1,
            "wall_ms": 1000.0 * (t1 - t0),
            "clip_ms": metrics.get("clip_ms"),
            "offload_ms": metrics.get("cache_offload_ms") or metrics.get("offload_ms"),
            "xfer_gate_ms": metrics.get("gate_xfer_ms") or metrics.get("xfer_gate_ms"),
            "gate_comp_ms": metrics.get("gate_compute_ms") or metrics.get("gate_comp_ms"),
            
            # 여기서 받아와야 통계가 잡힙니다!
            "mamba_ms": metrics.get("mamba_ms", 0.0),
            "trans_ms": metrics.get("trans_ms", 0.0),
        })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--fps_list", default="1,32,64")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--outdir", default="result/final_breakdown")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--model_base", default=None)
    ap.add_argument("--model_name", default="VideoLLaMA2-7B")
    ap.add_argument("--gate_modes", default="gpu,cpu")
    ap.add_argument("--cpu_dtype", default="bf16")
    ap.add_argument("--max_t", type=int, default=64)
    ap.add_argument("--gate_window", type=int, default=64)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("--instruct", default="Describe the video briefly.")
    args = ap.parse_args()

    os.environ["STREAMMIND_CPU_DTYPE"] = args.cpu_dtype
    os.environ["STREAMMIND_MAX_T"] = str(args.max_t)
    os.environ["STREAMMIND_GATE_WINDOW"] = str(args.gate_window)
    os.environ["STREAMMIND_BENCH_GATE_ONLY"] = "1"
    os.environ["STREAMMIND_SYNC_GATE"] = "1"

    try:
        model, processor, tokenizer, version = model_init(args.model_path, args.model_base, args.model_name)
    except TypeError:
        model, processor, tokenizer, version = model_init(args.model_path, args.model_name)

    vr = VideoReader(args.video, ctx=cpu(0), num_threads=1)

    fps_list = [int(x.strip()) for x in args.fps_list.split(",") if x.strip()]
    gate_modes = [x.strip().lower() for x in args.gate_modes.split(",") if x.strip()]

    rows = []
    for mode in gate_modes:
        if mode not in ("cpu", "gpu"):
            raise ValueError(f"bad gate mode: {mode}")

        os.environ["STREAMMIND_GATE_MODE"] = mode
        print(f"\n[BENCH] Running Mode={mode}, FPS List={fps_list}")

        for fps in fps_list:
            run_one_setting(model, processor, tokenizer, version, vr,
                            fps, args.steps, args.warmup, args.realtime,
                            args.instruct, mode, rows)

    # 1. Raw Data 저장 (필드 추가)
    raw_path = os.path.join(args.outdir, "raw.csv")
    fields = ["gate_mode","sampling_fps","frame_id","cls_pred","wall_ms",
              "clip_ms","offload_ms","xfer_gate_ms","gate_comp_ms", "mamba_ms", "trans_ms"]
    write_csv(raw_path, rows, fields)

    # 2. Summary 통계 저장
    summary = summarize(rows)
    sum_path = os.path.join(args.outdir, "summary.csv")
    if summary:
        write_csv(sum_path, summary, list(summary[0].keys()))

    # 3. 그래프 그리기
    title_suffix = (
        f"MAX_T={args.max_t}|W={args.gate_window}|dtype={args.cpu_dtype}|RT={int(args.realtime)}"
    )
    plot_summary(args.outdir, summary, title_suffix, logy=args.logy)

    print(f"[OK] Done. Results saved to {args.outdir}")


if __name__ == "__main__":
    main()