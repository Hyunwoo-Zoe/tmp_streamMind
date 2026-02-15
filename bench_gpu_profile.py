import torch
import time
import os
from streammind import model_init

def profile_gpu_resources(model, processor, video_tensor):
    print("\n" + "="*50)
    print(" GPU Resource Profiling (Scavenging Analysis)")
    print("="*50)
    
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 1. CLIP (Vision Tower) - GPU 유지 대상
        torch.cuda.synchronize()
        m1 = torch.cuda.memory_allocated()
        t1_start = torch.cuda.Event(enable_timing=True)
        t1_end = torch.cuda.Event(enable_timing=True)
        
        t1_start.record()
        with torch.no_grad():
            frames_features = model.get_model().get_vision_tower()(video_tensor)
            # Spatial Pooling (Mamba 입력용 준비)
            pooled_features = frames_features.mean(1).unsqueeze(1) 
        t1_end.record()
        torch.cuda.synchronize()
        
        clip_time = t1_start.elapsed_time(t1_end)
        clip_mem = (torch.cuda.memory_allocated() - m1) / 1024**2

        # 2. Mamba SSM - CPU 이관 시 이득 볼 구간
        m2 = torch.cuda.memory_allocated()
        t2_start = torch.cuda.Event(enable_timing=True)
        t2_end = torch.cuda.Event(enable_timing=True)
        
        t2_start.record()
        with torch.no_grad():
            # 실제 builder.py의 Video_Mamba_seq 로직 모사
            # pre_net -> mamba -> post_net
            x = model.get_model().mm_projector.pre_net(pooled_features)
            x = model.get_model().mm_projector.mamba_model(x)
            x_mamba_out = model.get_model().mm_projector.post_net(x)
        t2_end.record()
        torch.cuda.synchronize()
        
        mamba_time = t2_start.elapsed_time(t2_end)
        mamba_mem = (torch.cuda.memory_allocated() - m2) / 1024**2

        # 3. Gate (4-Layer Transformer) - CPU 이관 시 이득 볼 구간
        m3 = torch.cuda.memory_allocated()
        t3_start = torch.cuda.Event(enable_timing=True)
        t3_end = torch.cuda.Event(enable_timing=True)
        
        # Mistral Gate 입력 구조 맞춤: [Batch, Sequence, Hidden]
        # x_mamba_out에서 직접 입력을 시뮬레이션
        t3_start.record()
        with torch.no_grad():
            _ = model.get_model().mm_projector.cls_net(x_mamba_out, None, None)
        t3_end.record()
        torch.cuda.synchronize()
        
        gate_time = t3_start.elapsed_time(t3_end)
        gate_mem = (torch.cuda.memory_allocated() - m3) / 1024**2

        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        print(f"\n[Scavenging Potential Report]")
        print(f"1. GPU Base (Weights): {m1/1024**2:.2f} MB")
        print(f"2. CLIP (Keep on GPU): {clip_time:.2f}ms | +{clip_mem:.2f} MB")
        print(f"3. Mamba (Offloadable): {mamba_time:.2f}ms | +{mamba_mem:.2f} MB")
        print(f"4. Gate (Offloadable): {gate_time:.2f}ms | +{gate_mem:.2f} MB")
        print(f"\n=> Total GPU Time you can save per frame: {mamba_time + gate_time:.2f}ms")
        print(f"=> Total Peak Memory saved: {mamba_mem + gate_mem:.2f} MB (Activation only)")

    except Exception as e:
        print(f"\n[ERROR] Profiling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model_name = "VideoLLaMA2-7B"
    model, processor, tokenizer, version = model_init(model_path=None, model_name=model_name)
    
    # 336 해상도 (VideoLLaMA2 표준)
    dummy_video = torch.randn(1, 3, 336, 336).cuda().to(torch.bfloat16)
    profile_gpu_resources(model, processor, dummy_video)