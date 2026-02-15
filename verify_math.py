import torch
import os
import sys

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.append(os.getcwd())

from streammind.model.multimodal_projector.builder import Video_Mamba_seq, SSMConfig

def test_equivalence():
    print("=== [검증 1] Mamba O(T) vs O(1) 등가성 테스트 ===")
    
    torch.manual_seed(42) # 재현성을 위해 시드 고정
    
    # 1. 모델 초기화 (FP32로 정밀 검증)
    config = SSMConfig()
    config.d_model = 1024
    config.mm_hidden_size = 1024
    config.hidden_size = 4096
    config.n_ssm = 1
    
    print("[Info] 모델 빌드 중...")
    model = Video_Mamba_seq(config).float()
    model.eval()
    
    # 2. 가상의 비디오 입력 (Batch=1, Time=5, Tokens=1, Dim=1024)
    # T=5 프레임이 들어왔다고 가정
    T_seq = 5
    dummy_input = torch.randn(1, T_seq, 1, 1024)
    
    # -------------------------------------------------
    # CASE A: 기존 Forward (Ground Truth)
    # -------------------------------------------------
    print("[Info] Full Forward 실행 중...")
    with torch.no_grad():
        # Video_Mamba_seq.forward는 기본적으로 x를 반환
        out_full = model(dummy_input) 
        # out_full shape: [1, 5, 1024]
        
    # -------------------------------------------------
    # CASE B: Recurrent Step (Incremental)
    # -------------------------------------------------
    print("[Info] Recurrent Step 실행 중...")
    with torch.no_grad():
        state = None
        outputs_step = []
        
        for t in range(T_seq):
            # 한 프레임씩 짤라서 넣기
            x_step = dummy_input[:, t:t+1, :, :] # [1, 1, 1, 1024]
            
            # step 호출 (x_out, cls, state 반환)
            x_out, _, state = model.step(x_step, state=state)
            outputs_step.append(x_out)
            
        # 결과를 다시 합침: [1, 5, 1024]
        out_step_final = torch.cat(outputs_step, dim=1)

    # -------------------------------------------------
    # 비교 (Difference Check)
    # -------------------------------------------------
    # 전체 시퀀스에 대해 오차 측정
    max_diff = torch.abs(out_full - out_step_final).max()
    mean_diff = torch.abs(out_full - out_step_final).mean()
    
    print(f"\n[결과 분석]")
    print(f" - Max Difference : {max_diff.item():.8f}")
    print(f" - Mean Difference: {mean_diff.item():.8f}")
    
    # 허용 오차: FP32 기준 1e-5 이하, BF16 모델이면 1e-2 정도까지 허용
    threshold = 1e-4
    if max_diff < threshold:
        print("\n✅ [SUCCESS] 검증 성공! 두 방식의 결과가 수학적으로 일치합니다.")
    else:
        print("\n❌ [FAIL] 검증 실패. 수식이 다릅니다. ssm.py의 step 함수를 확인하세요.")
        print("Tip: unsqueeze 위치나 deltaA, deltaB 계산 순서를 확인해보세요.")

if __name__ == "__main__":
    test_equivalence()