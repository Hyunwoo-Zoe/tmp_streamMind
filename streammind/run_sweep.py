# streammind/run_sweep.py
import os
import csv
import time
import argparse
import random

import torch
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

from streammind import model_init
from streammind.conversation import conv_templates, SeparatorStyle
from streammind.mm_utils import tokenizer_MMODAL_token, KeywordsStoppingCriteria, expand2square
from streammind.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX


def reset_stream_state(model):
    # K별로 동일 초기조건을 위해 반드시 reset
    if hasattr(model, "frame_feature"):
        model.frame_feature = None
    if hasattr(model, "past_review_caption"):
        model.past_review_caption = None
    if hasattr(model, "interval_id_list"):
        model.interval_id_list = []


def build_prompt(tokenizer, version, question: str):
    modal_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_index = MMODAL_TOKEN_INDEX["VIDEO"]

    instruct = modal_token + "\n" + question

    conv = conv_templates[version].copy()
    conv.append_message(conv.roles[0], instruct)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_index, return_tensors="pt").unsqueeze(0)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
    stopping_criteria = [KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)]

    return input_ids, attention_mask, stopping_criteria


def get_processor(model):
    vision = model.get_model().get_vision_tower()
    return vision.image_processor


def make_video_tensor(vr, processor, frame_ids):
    arr = vr.get_batch(frame_ids).asnumpy()  # (t,h,w,c)
    images = [Image.fromarray(f) for f in arr]
    images = [expand2square(img, tuple(int(x * 255) for x in processor.image_mean)) for img in images]
    video_tensor = processor.preprocess(images, return_tensors="pt")["pixel_values"]  # (t,3,336,336)
    return video_tensor


def measure_one_call(model, tokenizer, input_ids, attention_mask, stopping_criteria, video_tensor, max_new_tokens, only_cls: bool):
    # wall time
    t0 = time.perf_counter()

    # GPU active time (CUDA event) : nvidia-smi util 대신 이걸로 “GPU가 실제로 바빴는지”를 수치화
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()

    out = model.stream_generate(
        inputs=input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        images_or_videos=[video_tensor.half().cuda()],
        tokenizer=tokenizer,
        do_sample=False,
        temperature=0.0,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        only_cls=only_cls,
        return_cls=False,
    )

    if use_cuda:
        ev1.record()
        torch.cuda.synchronize()
        gpu_ms = ev0.elapsed_time(ev1)
    else:
        gpu_ms = 0.0

    wall_ms = (time.perf_counter() - t0) * 1000.0
    gpu_util_pct = (gpu_ms / wall_ms * 100.0) if wall_ms > 0 else 0.0

    text_len = 0
    if isinstance(out, str):
        text_len = len(out)

    return wall_ms, gpu_ms, gpu_util_pct, out, text_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", default="VideoLLaMA2-7B")
    ap.add_argument("--video", default="assets/blind_com_demo.mp4")
    ap.add_argument("--question", default="Describe what is happening now in one short phrase.")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--k-list", required=True, help="e.g. 64,128,256,512")
    ap.add_argument("--prefill-steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # 재현성(필수 수준만)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # 모델 로드
    model, _, tokenizer, version = model_init(model_path=args.model_path, model_name=args.model_name)
    model.eval()

    # 프롬프트/스톱 조건 준비
    input_ids, attention_mask, stopping_criteria = build_prompt(tokenizer, version, args.question)

    # 비디오 준비
    vr = VideoReader(args.video, ctx=cpu(0), num_threads=1)
    total = len(vr)
    fps = float(vr.get_avg_fps())
    processor = get_processor(model)

    # 스윕 리스트
    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]

    # CSV 헤더
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k",
            "iter",
            "wall_ms",
            "gpu_ms",
            "gpu_util_pct",
            "text_len",
            "chunk_start_frame",
            "chunk_end_frame",
            "ts0_sec",
            "ts1_sec",
        ])

        # 워밍업(버림)
        for _ in range(args.warmup):
            os.environ["STREAMMIND_K"] = str(k_list[0])
            reset_stream_state(model)

            # prefill (decode 금지)
            for s in range(args.prefill_steps):
                start = (s * args.chunk) % max(1, total)
                end = min(start + args.chunk, total)
                frame_ids = list(range(start, end))
                vt = make_video_tensor(vr, processor, frame_ids)
                measure_one_call(model, tokenizer, input_ids, attention_mask, stopping_criteria, vt, args.max_new_tokens, only_cls=True)

            # measure 1회(버림)
            start = (args.prefill_steps * args.chunk) % max(1, total)
            end = min(start + args.chunk, total)
            frame_ids = list(range(start, end))
            vt = make_video_tensor(vr, processor, frame_ids)
            measure_one_call(model, tokenizer, input_ids, attention_mask, stopping_criteria, vt, args.max_new_tokens, only_cls=False)

        # 본 측정
        for k in k_list:
            os.environ["STREAMMIND_K"] = str(k)
            reset_stream_state(model)

            # prefill: cache를 채우는 단계 (decode 금지)
            for s in range(args.prefill_steps):
                start = (s * args.chunk) % max(1, total)
                end = min(start + args.chunk, total)
                frame_ids = list(range(start, end))
                vt = make_video_tensor(vr, processor, frame_ids)
                measure_one_call(model, tokenizer, input_ids, attention_mask, stopping_criteria, vt, args.max_new_tokens, only_cls=True)

            # measure: 동일 chunk로 반복 측정 (K만 바뀐 상태)
            m_start = (args.prefill_steps * args.chunk) % max(1, total)
            m_end = min(m_start + args.chunk, total)
            frame_ids = list(range(m_start, m_end))
            vt = make_video_tensor(vr, processor, frame_ids)

            ts0 = m_start / fps
            ts1 = m_end / fps

            for it in range(args.iters):
                wall_ms, gpu_ms, gpu_util_pct, out, text_len = measure_one_call(
                    model, tokenizer, input_ids, attention_mask, stopping_criteria, vt, args.max_new_tokens, only_cls=False
                )
                w.writerow([k, it, f"{wall_ms:.4f}", f"{gpu_ms:.4f}", f"{gpu_util_pct:.2f}", text_len, m_start, m_end, f"{ts0:.4f}", f"{ts1:.4f}"])
                f.flush()

    print(f"[OK] wrote: {args.out}")


if __name__ == "__main__":
    main()
