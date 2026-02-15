# streammind/bench_llm_prefill_scale.py
# LLM Transformer Prefill Scalability Benchmark (GPU vs CPU AMX/IPEX)

import os
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import gc

from streammind import model_init

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="VideoLLaMA2-7B")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--cpu_dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--max_tokens", type=int, default=78000, help="Max sequence length")
    parser.add_argument("--step_tokens", type=int, default=8192, help="Step size")
    parser.add_argument("--outdir", default="result/llm_prefill_test")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    # AMX 활성화
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_AMX"
    
    print(f"=== [LLM Prefill Benchmark] ===")
    print(f"Model: {args.model_name}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"CPU Dtype: {args.cpu_dtype}")
    print(f"===============================\n")

    # 토큰 길이 리스트 생성 (1k 부터 시작)
    token_counts = [1024] + list(range(args.step_tokens, args.max_tokens + 1, args.step_tokens))
    
    gpu_results = {}
    cpu_results = {}

    # ----------------------------------------------------------------
    # 1. GPU Measurement Loop
    # ----------------------------------------------------------------
    print("\n[Phase 1] Measuring GPU Performance...")
    try:
        # 모델 로드 (GPU)
        model, _, _, _ = model_init(args.model_path, args.model_name)
        model = model.cuda().eval()
        
        for seq_len in tqdm(token_counts, desc="GPU Prefill"):
            try:
                # Dummy Input (Batch=1)
                input_ids = torch.randint(0, 32000, (1, seq_len)).cuda()
                
                # Warmup
                with torch.inference_mode():
                    _ = model(input_ids, use_cache=True)
                torch.cuda.synchronize()
                
                # Measure
                t0 = time.perf_counter()
                with torch.inference_mode():
                    # use_cache=True를 켜야 KV Cache를 생성하는 Prefill 동작을 함
                    _ = model(input_ids, use_cache=True)
                torch.cuda.synchronize()
                
                latency = (time.perf_counter() - t0) * 1000.0
                gpu_results[seq_len] = latency
                
            except torch.cuda.OutOfMemoryError:
                print(f"[GPU OOM] Failed at {seq_len} tokens")
                gpu_results[seq_len] = None
                torch.cuda.empty_cache()
                break
                
        del model
        del input_ids
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"[GPU Error] {e}")

    # ----------------------------------------------------------------
    # 2. CPU Measurement Loop (with IPEX)
    # ----------------------------------------------------------------
    print("\n[Phase 2] Measuring CPU (AMX+IPEX) Performance...")
    try:
        # 모델 다시 로드 (CPU)
        # model_init은 기본적으로 GPU로 올리려 할 수 있으므로 cpu로 강제 이동 필요
        model, _, _, _ = model_init(args.model_path, args.model_name)
        model = model.cpu().eval()
        
        # IPEX 최적화
        import intel_extension_for_pytorch as ipex
        #cpu_dtype = torch.bfloat16 if args.cpu_dtype == "bf16" else torch.float32
        
        print("[INFO] Applying IPEX optimization for LLM...")
        # LLM 전체(Transformer)를 최적화
        model = ipex.optimize(model, dtype=cpu_dtype, inplace=True)
        
        # Torch Compile (선택사항: IPEX만으로도 충분히 빠르지만 시도 가능)
        # model = torch.compile(model, backend="ipex", dynamic=True)

        for seq_len in tqdm(token_counts, desc="CPU Prefill"):
            try:
                input_ids = torch.randint(0, 32000, (1, seq_len)).cpu()
                
                # Warmup
                with torch.inference_mode(), torch.cpu.amp.autocast(enabled=True, dtype=cpu_dtype):
                    _ = model(input_ids, use_cache=True)
                
                # Measure
                t0 = time.perf_counter()
                with torch.inference_mode(), torch.cpu.amp.autocast(enabled=True, dtype=cpu_dtype):
                    _ = model(input_ids, use_cache=True)
                
                latency = (time.perf_counter() - t0) * 1000.0
                cpu_results[seq_len] = latency
                
            except Exception as e:
                print(f"[CPU Error at {seq_len}] {e}")
                cpu_results[seq_len] = None

    except Exception as e:
        print(f"[CPU Setup Error] {e}")

    # ----------------------------------------------------------------
    # 3. Save & Plot
    # ----------------------------------------------------------------
    data = []
    for t in token_counts:
        g = gpu_results.get(t)
        c = cpu_results.get(t)
        data.append({
            "tokens": t,
            "gpu_ms": g,
            "cpu_ms": c,
            "speedup_gpu_over_cpu": (c / g) if (c and g) else None
        })
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(args.outdir, "llm_prefill_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    plot_results(df, args.outdir, args.cpu_dtype)

def plot_results(df, outdir, dtype):
    plt.figure(figsize=(12, 7))
    
    # Valid data filtering
    valid_gpu = df.dropna(subset=["gpu_ms"])
    valid_cpu = df.dropna(subset=["cpu_ms"])
    
    plt.plot(valid_gpu["tokens"], valid_gpu["gpu_ms"], label="GPU (A100/H100)", marker="o", linewidth=2, color="#1f77b4")
    plt.plot(valid_cpu["tokens"], valid_cpu["cpu_ms"], label=f"CPU (AMX+IPEX {dtype})", marker="s", linewidth=2, color="#ff7f0e")
    
    plt.title(f"LLM Transformer Prefill Latency (GPU vs CPU)")
    plt.xlabel("Input Sequence Length (Tokens)")
    plt.ylabel("Latency (ms)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    # 값 표시 (드문드문)
    for i, row in valid_cpu.iterrows():
        if i % 2 == 0: 
            plt.annotate(f"{row['cpu_ms']:.0f}", (row['tokens'], row['cpu_ms']), 
                         xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8, color="#ff7f0e")
    
    for i, row in valid_gpu.iterrows():
        if i % 2 == 0:
            plt.annotate(f"{row['gpu_ms']:.0f}", (row['tokens'], row['gpu_ms']), 
                         xytext=(0, -15), textcoords='offset points', ha='center', fontsize=8, color="#1f77b4")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "llm_prefill_comparison.png"))
    print(f"Saved plot to {os.path.join(outdir, 'llm_prefill_comparison.png')}")

if __name__ == "__main__":
    run_benchmark()