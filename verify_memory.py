import torch
import os
import sys
import psutil
import time

sys.path.append(os.getcwd())
from streammind.model.multimodal_projector.builder import Video_Mamba_seq, SSMConfig

def test_stability():
    print("=== [검증 2] 장기 실행 메모리 안정성 테스트 (10,000 Steps) ===")
    
    config = SSMConfig()
    config.d_model = 1024
    config.mm_hidden_size = 1024
    config.hidden_size = 4096 
    config.n_ssm = 1

    model = Video_Mamba_seq(config).cuda() # GPU 환경 가정 (없으면 .cpu())
    model.eval()
    
    # 더미 입력
    x = torch.randn(1, 1, 1, 1024).cuda()
    state = None
    
    process = psutil.Process()
    start_mem = process.memory_info().rss / 1024 / 1024  # MB 단위
    print(f"[Start] 초기 메모리: {start_mem:.2f} MB")
    
    # 10,000 프레임 처리 루프
    for i in range(1, 10001):
        with torch.no_grad():
            _, _, state = model.step(x, state=state)
        
        # 1000 프레임마다 체크
        if i % 1000 == 0:
            curr_mem = process.memory_info().rss / 1024 / 1024
            delta = curr_mem - start_mem
            print(f"Step {i:5d}: {curr_mem:.2f} MB (변화량: {delta:+.2f} MB)")
            
            # 500MB 이상 늘어나면 누수로 간주 (엄격한 기준)
            if delta > 500: 
                print("\n❌ [FAIL] 메모리 누수 감지! state 변수에 detach()가 필요한지 확인하세요.")
                return

    print("\n✅ [SUCCESS] 메모리 안정성 확인 완료.")

if __name__ == "__main__":
    test_stability()