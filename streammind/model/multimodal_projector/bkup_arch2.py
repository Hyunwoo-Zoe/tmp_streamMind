from typing import Optional
import os
import contextlib
from abc import ABC, abstractmethod
import random
import einops
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import copy
from .multimodal_projector import load_mm_projector
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from ..mm_utils import get_anyres_image_grid_shape
from ..constants import NUM_FRAMES, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,DEFAULT_MMODAL_PATCH_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from itertools import accumulate

class Videollama2MetaModel:

    def __init__(self, config):
        super(Videollama2MetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                if os.path.isdir(pretrain_mm_mlp_adapter):
                    mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
                else:
                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


class Videollama2MetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images_or_videos(self, images_or_videos, modalities):
        num_frames = self.config.num_frames if hasattr(self.config, 'num_frames') else NUM_FRAMES

        videos = [x.unsqueeze(0).expand(num_frames, -1, -1, -1) if modal == 'image' else x for x, modal in zip(images_or_videos, modalities)]
        videos = torch.stack(videos, dim=0)

        assert len(videos.size()) == 5
        batch_size = videos.size(0)

        frames = einops.rearrange(videos, 'b t c h w -> (b t) c h w')
        frames_features = self.get_model().get_vision_tower()(frames)
        frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)

        return self.temporal_aggregator(frames_features)

    def encode_images_or_videos_score_cls_inference_allframe(
        self,
        images_or_videos,
        past_frames_features: Optional[torch.Tensor] = None,
        frames_features_shape: Optional[list] = None,
    ):
        """
        STREAMMIND_GATE_MODE="gpu" : CLIP(GPU) -> Mamba+Gate(GPU)
        STREAMMIND_GATE_MODE="cpu" : CLIP(GPU) -> (D2H offload) -> Mamba+Gate(CPU BF16/AMX)
        """
        gate_mode = os.getenv("STREAMMIND_GATE_MODE", "gpu").strip().lower()  # "gpu" | "cpu"
        if gate_mode not in ("gpu", "cpu"):
            gate_mode = "gpu"

        cpu_dtype_env = os.getenv("STREAMMIND_CPU_DTYPE", "bf16").strip().lower()
        cpu_dtype = torch.bfloat16 if cpu_dtype_env in ("bf16", "bfloat16") else torch.float32

        # -----------------------------
        # input normalize
        # -----------------------------
        if isinstance(images_or_videos, (list, tuple)):
            if len(images_or_videos) != 1:
                raise ValueError(f"Expected 1 item in list, got {len(images_or_videos)}")
            images_or_videos = images_or_videos[0]

        if not hasattr(images_or_videos, "shape"):
            raise TypeError(f"images_or_videos must be a torch.Tensor, got {type(images_or_videos)}")

        videos = images_or_videos.unsqueeze(0)  # [1, T, C, H, W]
        assert len(videos.size()) == 5
        batch_size = videos.size(0)

        frames = einops.rearrange(videos, "b t c h w -> (b t) c h w")
        if frames.shape[0] > 600:
            frames = frames[-600:]

        def _mb(x: torch.Tensor) -> float:
            return x.numel() * x.element_size() / (1024 * 1024)

        # -----------------------------
        # 1) CLIP feature (usually GPU)
        # -----------------------------
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_clip0 = time.perf_counter()
        frames_features = self.get_model().get_vision_tower()(frames)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        clip_ms = 1000.0 * (time.perf_counter() - t_clip0)

        # [B, T, N, H]
        frames_features = einops.rearrange(frames_features, "(b t) n h -> b t n h", b=batch_size)

        # -----------------------------
        # 2) gate mode routing: GPU keep / CPU offload(BF16)
        # -----------------------------
        offload_ms = 0.0
        xfer_past_ms = 0.0

        if gate_mode == "cpu":
            if frames_features.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_off0 = time.perf_counter()
            frames_features = frames_features.detach().to("cpu", dtype=cpu_dtype).contiguous()
            # CPU에서만 쓸거면 pin_memory는 필수 아님. (H2D가 잦으면 의미 있음)
            # frames_features = frames_features.pin_memory()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            offload_ms += 1000.0 * (time.perf_counter() - t_off0)
        else:
            frames_features = frames_features.detach()

        # -----------------------------
        # 3) past concat (device 맞추기)
        # -----------------------------
        if past_frames_features is not None:
            past_frames_features = past_frames_features.detach()
            if past_frames_features.device != frames_features.device:
                if gate_mode == "cpu":
                    if past_frames_features.device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_p0 = time.perf_counter()
                    past_frames_features = past_frames_features.to("cpu", dtype=cpu_dtype).contiguous()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    xfer_past_ms = 1000.0 * (time.perf_counter() - t_p0)
                    offload_ms += xfer_past_ms
                else:
                    t_p0 = time.perf_counter()
                    past_frames_features = past_frames_features.to(frames_features.device, non_blocking=True)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    xfer_past_ms = 1000.0 * (time.perf_counter() - t_p0)

            frames_features = torch.cat((past_frames_features, frames_features), dim=1)

        # -----------------------------
        # 4) MAX_T bookkeeping (원래 로직 유지)
        # -----------------------------
        MAX_T = int(os.environ.get("STREAMMIND_MAX_T", "64"))
        cur_total_t = int(frames_features.shape[1])

        if cur_total_t > MAX_T:
            offset = cur_total_t - MAX_T
            frames_features = frames_features[:, -MAX_T:].contiguous()

            if frames_features_shape is None:
                frames_features_shape = []
            new_shape = []
            for x in frames_features_shape:
                x = int(x)
                if x > offset:
                    new_shape.append(x - offset)

            if len(new_shape) == 0 or new_shape[-1] != MAX_T:
                new_shape.append(MAX_T)
            frames_features_shape = new_shape
        else:
            if frames_features_shape is None:
                frames_features_shape = []
            if len(frames_features_shape) == 0 or int(frames_features_shape[-1]) != cur_total_t:
                frames_features_shape.append(cur_total_t)

        # -----------------------------
        # 5) GATE_WINDOW
        # -----------------------------
        W = int(os.environ.get("STREAMMIND_GATE_WINDOW", "64"))
        frames_for_gate = frames_features
        frames_features_shape_for_gate = frames_features_shape

        if W > 0 and frames_features.shape[1] > W:
            frames_for_gate = frames_features[:, -W:]
            frames_features_shape_for_gate = [
                min(x, W)
                for x in frames_features_shape
                if x > (frames_features.shape[1] - W)
            ]
            if len(frames_features_shape_for_gate) == 0:
                frames_features_shape_for_gate = [W]

        # [수정됨] 들여쓰기 교정 (def는 4칸, 본문은 8칸)
    # [수정됨] NameError 해결 및 시간 측정 로직 개선
    def encode_images_or_videos_score_cls_inference_allframe(
        self,
        images_or_videos,
        past_frames_features: Optional[torch.Tensor] = None,
        frames_features_shape: Optional[list] = None,
    ):
        # 1. 환경 변수 및 설정 로딩
        gate_mode = os.getenv("STREAMMIND_GATE_MODE", "gpu").strip().lower()
        if gate_mode not in ("gpu", "cpu"):
            gate_mode = "gpu"

        cpu_dtype_env = os.getenv("STREAMMIND_CPU_DTYPE", "bf16").strip().lower()
        cpu_dtype = torch.bfloat16 if cpu_dtype_env in ("bf16", "bfloat16") else torch.float32

        # 2. 디버깅 변수 초기화 (여기가 중요합니다)
        self._dbg_mamba_ms = 0.0
        self._dbg_trans_ms = 0.0
        
        # 3. 입력 데이터 전처리
        if isinstance(images_or_videos, (list, tuple)):
            if len(images_or_videos) != 1:
                raise ValueError(f"Expected 1 item in list, got {len(images_or_videos)}")
            images_or_videos = images_or_videos[0]

        if not hasattr(images_or_videos, "shape"):
            raise TypeError(f"images_or_videos must be a torch.Tensor, got {type(images_or_videos)}")

        videos = images_or_videos.unsqueeze(0)
        assert len(videos.size()) == 5
        batch_size = videos.size(0)

        frames = einops.rearrange(videos, "b t c h w -> (b t) c h w")
        if frames.shape[0] > 600:
            frames = frames[-600:]

        def _mb(x: torch.Tensor) -> float:
            return x.numel() * x.element_size() / (1024 * 1024)

        # 4. Vision Encoder (CLIP) 실행
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_clip0 = time.perf_counter()
        frames_features = self.get_model().get_vision_tower()(frames)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        clip_ms = 1000.0 * (time.perf_counter() - t_clip0)

        frames_features = einops.rearrange(frames_features, "(b t) n h -> b t n h", b=batch_size)

        # 5. Offload / Transfer 로직
        offload_ms = 0.0
        xfer_past_ms = 0.0

        if gate_mode == "cpu":
            if frames_features.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_off0 = time.perf_counter()
            frames_features = frames_features.detach().to("cpu", dtype=cpu_dtype).contiguous()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            offload_ms += 1000.0 * (time.perf_counter() - t_off0)
        else:
            frames_features = frames_features.detach()

        if past_frames_features is not None:
            past_frames_features = past_frames_features.detach()
            if past_frames_features.device != frames_features.device:
                if gate_mode == "cpu":
                    if past_frames_features.device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_p0 = time.perf_counter()
                    past_frames_features = past_frames_features.to("cpu", dtype=cpu_dtype).contiguous()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    xfer_past_ms = 1000.0 * (time.perf_counter() - t_p0)
                    offload_ms += xfer_past_ms
                else:
                    t_p0 = time.perf_counter()
                    past_frames_features = past_frames_features.to(frames_features.device, non_blocking=True)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    xfer_past_ms = 1000.0 * (time.perf_counter() - t_p0)

            frames_features = torch.cat((past_frames_features, frames_features), dim=1)

        # 6. Windowing & MAX_T 처리
        MAX_T = int(os.environ.get("STREAMMIND_MAX_T", "64"))
        cur_total_t = int(frames_features.shape[1])

        if cur_total_t > MAX_T:
            offset = cur_total_t - MAX_T
            frames_features = frames_features[:, -MAX_T:].contiguous()

            if frames_features_shape is None:
                frames_features_shape = []
            new_shape = []
            for x in frames_features_shape:
                x = int(x)
                if x > offset:
                    new_shape.append(x - offset)

            if len(new_shape) == 0 or new_shape[-1] != MAX_T:
                new_shape.append(MAX_T)
            frames_features_shape = new_shape
        else:
            if frames_features_shape is None:
                frames_features_shape = []
            if len(frames_features_shape) == 0 or int(frames_features_shape[-1]) != cur_total_t:
                frames_features_shape.append(cur_total_t)

        W = int(os.environ.get("STREAMMIND_GATE_WINDOW", "64"))
        frames_for_gate = frames_features
        frames_features_shape_for_gate = frames_features_shape

        if W > 0 and frames_features.shape[1] > W:
            frames_for_gate = frames_features[:, -W:]
            frames_features_shape_for_gate = [
                min(x, W)
                for x in frames_features_shape
                if x > (frames_features.shape[1] - W)
            ]
            if len(frames_features_shape_for_gate) == 0:
                frames_features_shape_for_gate = [W]

        # 7. CPU Projector 빌드 및 최적화 (IPEX)
        if gate_mode == "cpu":
            gate_dev = torch.device("cpu")
            need_build = (not hasattr(self, "_mm_projector_cpu")) or (self._mm_projector_cpu is None)
            if need_build or getattr(self, "_mm_projector_cpu_dtype", None) != cpu_dtype:
                print(f"[INIT] Building CPU Projector (Dtype={cpu_dtype})...")
                self._mm_projector_cpu = copy.deepcopy(self.get_model().mm_projector).to("cpu", dtype=cpu_dtype)
                self._mm_projector_cpu.eval()
                for p in self._mm_projector_cpu.parameters():
                    p.requires_grad_(False)
                self._mm_projector_cpu_dtype = cpu_dtype

                try:
                    import intel_extension_for_pytorch as ipex
                    self._mm_projector_cpu = ipex.optimize(
                        self._mm_projector_cpu, 
                        dtype=cpu_dtype, 
                        inplace=True, 
                        weights_prepack=False,
                        graph_mode=True
                    )
                    print("[INFO] IPEX Optimization applied (weights_prepack=True).")
                    
                    # torch.compile (Inductor 백엔드 사용 시 Mamba Scan 가속에 유리)
                    print("[INFO] Compiling CPU projector with torch.compile...")
                    self._mm_projector_cpu = torch.compile(self._mm_projector_cpu, backend="ipex")
                    
                    # Warmup
                    print("[INFO] Warming up CPU Projector...")
                    with torch.inference_mode(), torch.cpu.amp.autocast(enabled=True, dtype=cpu_dtype):
                        dummy_dim = self.config.mm_hidden_size
                        dummy_in = torch.randn(1, 10, dummy_dim, device="cpu", dtype=cpu_dtype)
                        proj_type = getattr(self.config, "mm_projector_type", "")
                        
                        if "mamba" in proj_type or "connector" in proj_type:
                            self._mm_projector_cpu(dummy_in, cls_inference=True, frames_features_shape=[10])
                        else:
                            self._mm_projector_cpu(dummy_in.mean(1))
                    print("[INFO] Warmup done.")
                    
                except Exception as e:
                    print(f"[WARN] Optimization/Warmup failed: {e}")

            mm_proj = self._mm_projector_cpu
        else:
            mm_proj = self.get_model().mm_projector
            gate_dev = mm_proj.parameters().__next__().device

        # 8. Gate Input Transfer Check
        xfer_gate_ms = 0.0
        frames_for_gate_dev = frames_for_gate

        if frames_for_gate_dev.device != gate_dev:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_x0 = time.perf_counter()
            frames_for_gate_dev = frames_for_gate_dev.to(gate_dev, non_blocking=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            xfer_gate_ms = 1000.0 * (time.perf_counter() - t_x0)

        # 9. Run Projector (Mamba + Gate) - 측정 로직 포함
        def _run_projector(frames_feat_dev: torch.Tensor):
            proj_type = getattr(self.config, "mm_projector_type", "")
            
            # [측정 시작]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            
            if proj_type in ("mlp2x_gelu", "linear"):
                out = mm_proj(frames_feat_dev.mean(1))
                # 일반 MLP는 Mamba/Gate가 없으므로 전체 시간을 기록
                self._dbg_mamba_ms = 1000.0 * (time.perf_counter() - t_start)
                return out

            elif proj_type in ("spatial_conv", "spatial_pool"):
                out = mm_proj(frames_feat_dev)
                self._dbg_mamba_ms = 1000.0 * (time.perf_counter() - t_start)
                return out

            elif ("tc_connector" in proj_type) or ("tp_connector" in proj_type) or ("mamba" in proj_type):
                # 1. Projector 실행 (내부에서 last_mamba_ms, last_gate_ms가 계산됨)
                out = mm_proj(
                    frames_feat_dev,
                    cls_inference=True,
                    cls_training=False,
                    cls_demo=False,
                    frames_features_shape=frames_features_shape_for_gate,
                )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # 2. builder.py에 저장된 개별 연산 시간 낚아채기
                # torch.compile 사용 시를 대비해 orig_mod 확인
                target_mod = mm_proj.orig_mod if hasattr(mm_proj, "orig_mod") else mm_proj
                
                mamba_ms = getattr(target_mod, "last_mamba_ms", 0.0)
                gate_ms = getattr(target_mod, "last_gate_ms", 0.0)
                
                # 3. 클래스 멤버 변수에 할당 (디버깅용)
                self._dbg_mamba_ms = float(mamba_ms)
                self._dbg_trans_ms = float(gate_ms) # Gate(Transformer) 시간을 여기에 저장
                
                return out
            else:
                raise Exception(f"Unsupported projector type {proj_type}!!!")

        # --- (앞서 정의한 _run_projector 호출 구간) ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_g0 = time.perf_counter()

        if gate_dev.type == "cpu":  # 오타 수정: "cpu`" -> "cpu"
            # [최적화] 스레드 수 조절 (Context Switching 방지)
            prev_threads = torch.get_num_threads()
            target_threads = int(os.environ.get("OMP_NUM_THREADS", "16")) 
            torch.set_num_threads(target_threads) 

            try:
                with torch.inference_mode(), torch.cpu.amp.autocast(enabled=True, dtype=cpu_dtype):
                    out = _run_projector(frames_for_gate_dev)
            finally:
                torch.set_num_threads(prev_threads)
        else:
            out = _run_projector(frames_for_gate_dev)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gate_compute_ms = 1000.0 * (time.perf_counter() - t_g0)

        # -----------------------------------------------------------
        # 10. 결과 언패킹 및 builder.py 상세 시간 취득
        # -----------------------------------------------------------
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            X_features, cls_feature = out[0], out[1]
        else:
            X_features, cls_feature = out, None

        frames_features_cache = frames_features.detach()

        # builder.py의 인스턴스에서 직접 개별 시간 가져오기
        target_mod = mm_proj.orig_mod if hasattr(mm_proj, "orig_mod") else mm_proj
        mamba_ms = getattr(target_mod, "last_mamba_ms", 0.0)
        gate_ms = getattr(target_mod, "last_gate_ms", 0.0)
        
        # 전체 합산 레이턴시 계산
        total_ms = clip_ms + offload_ms + xfer_gate_ms + mamba_ms + gate_ms

        # -----------------------------------------------------------
        # 11. 디버깅 필드 업데이트 및 최종 출력
        # -----------------------------------------------------------
        self._dbg_cpu_dtype = str(cpu_dtype)
        self._dbg_clip_ms = float(clip_ms)
        self._dbg_offload_ms = float(offload_ms)
        self._dbg_xfer_gate_ms = float(xfer_gate_ms)
        self._dbg_gate_compute_ms = float(gate_compute_ms)
        self._dbg_mamba_ms = float(mamba_ms)  # 상세 Mamba 시간 저장
        self._dbg_trans_ms = float(gate_ms)   # 상세 Gate 시간 저장
        self._dbg_gate_mode = gate_mode

        # 터미널 가독성을 위한 상세 리포트 출력
        print(f"\n" + "="*65)
        print(f" [STREAMMIND LATENCY REPORT]")
        print(f" 1. Vision (CLIP)      : {clip_ms:10.2f} ms")
        print(f" 2. Offload/Xfer       : {(offload_ms + xfer_gate_ms):10.2f} ms")
        print(f" 3. Projector (Mamba)  : {mamba_ms:10.2f} ms")
        print(f" 4. Gate (Mistral)     : {gate_ms:10.2f} ms")
        print("-" * 65)
        print(f" >> TOTAL STEP LATENCY : {total_ms:10.2f} ms")
        print(f" (Gate_Total_Compute   : {gate_compute_ms:10.2f} ms)") # 오버헤드 포함 시간
        print("="*65 + "\n")

        return X_features, cls_feature, frames_features_cache

    def encode_images_or_videos_score_cls_video_cls_autoregressive(self, images_or_videos,cls_inference = False,cls_training = False,caption_info = None,prompt_time_input_ids = None,prompt_time_lable = None):
        frames_features_list = []
        frames_features_shape = []
        for idx, images_or_video in enumerate(images_or_videos):
            num_frames = images_or_video.shape[0]
            videos = images_or_video.unsqueeze(0)

            assert len(videos.size()) == 5
            batch_size = videos.size(0)

            frames = einops.rearrange(videos, 'b t c h w -> (b t) c h w')

            if frames.shape[0] > 600:
                frames = frames[-600:]
            frames_features = self.get_model().get_vision_tower()(frames)
            frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)
            frames_features_list.append(frames_features)
            frames_features_shape.append(frames_features.shape[1])

        frames_features_shape = list(accumulate(frames_features_shape))

        frames_features = torch.cat(frames_features_list,dim=1)

        exactor_output = self.temporal_aggregator(frames_features,
                                cls_inference = cls_inference,cls_training = cls_training,
                                frames_features_shape = frames_features_shape,
                                prompt_time_input_ids = prompt_time_input_ids,
                                prompt_time_lable = prompt_time_lable)

        return exactor_output, frames_features_shape

    def encode_images_or_videos_score_cls_inference_allframe_demo(self, images_or_videos, past_frames_features, frames_features_shape):
        num_frames = images_or_videos.shape[0]
        videos = images_or_videos.unsqueeze(0)

        assert len(videos.size()) == 5
        batch_size = videos.size(0)

        frames = einops.rearrange(videos, 'b t c h w -> (b t) c h w')
        if frames.shape[0] > 600:
            frames = frames[-600:]
        frames_features = self.get_model().get_vision_tower()(frames)
        frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)
        if past_frames_features is not None:
            frames_features = torch.cat((past_frames_features, frames_features), dim = 1)

        interval_id = frames_features.shape[1]
        X_features, cls_feature = self.temporal_aggregator(frames_features, 
                                                        cls_inference = False, cls_training = False, cls_demo = True,
                                                        frames_features_shape = frames_features_shape)
        return X_features, cls_feature,frames_features,interval_id

    def mamba_encode_images_or_videos_score(self, frames_features):
        return self.temporal_aggregator(frames_features)

    @torch.no_grad()
    def encode_all_videos_score(self, images_or_videos, modalities):
        def find_video_files(root_path, target_filenames):
            paths = []
            for dirpath, _, filenames in os.walk(root_path):
                for target_filename in target_filenames:
                    if target_filename in filenames:
                        paths.append(os.path.join(dirpath, target_filename))
            return paths
        from decord import VideoReader, cpu
        from PIL import Image
        from videollama2.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image, process_score_video

        import torch.distributed as dist
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        processor = self.get_model().get_vision_tower().image_processor

        target_filenames = ["1_224p.mkv", "2_224p.mkv"]

        score_video_list = find_video_files("/mnt/input/MatchTime/features_video",target_filenames)
        local_batch = len(score_video_list)// world_size

        for video_path in score_video_list[rank * local_batch : (rank+1) * local_batch]:
            print("################encode video ########################")
            print(video_path)
            file_name = os.path.basename(video_path)
            half = file_name.split("_224p.mkv")[0]
            if isinstance(video_path, str):
                decord_vr = VideoReader(uri=video_path, ctx=cpu(0), num_threads=1) 
                duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())

                for start in range(0 , duration , 500):                    
                    frame_id_list = list(range(start , start + 500))
                    if start + 500 > duration:
                        frame_id_list = list(range(start ,duration))

                    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
                    

                    images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
                    images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
                    video = processor.preprocess(images, return_tensors='pt')['pixel_values']

                    num_frames = video.shape[0]
                    videos = video.unsqueeze(0)

                    assert len(videos.size()) == 5
                    batch_size = videos.size(0)

                    frames = einops.rearrange(videos, 'b t c h w -> (b t) c h w')
                    frames_features = self.get_model().get_vision_tower()(frames)
                    frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)

                    encode_feature_new_path = video_path.replace("features_video", "features_video_encode_ddp")
                    encode_feature_new_path = os.path.dirname(encode_feature_new_path)
                    encode_feature_new_path = os.path.join(encode_feature_new_path, "{}_encode_feature_frame_{}_{}.pt".format(half,start,start+500))
                    print("save feature to {}".format(encode_feature_new_path))

    def temporal_aggregator(self, frames_features,cls_demo = False,cls_inference = False,cls_training = False,caption_info = None,frames_features_shape=None,tokenizer = None,prompt_time_input_ids = None,prompt_time_lable = None):
        if self.config.mm_projector_type == "mlp2x_gelu" or self.config.mm_projector_type == "linear":
            video_features = self.get_model().mm_projector(frames_features.mean(1))
        elif self.config.mm_projector_type == "spatial_conv":
            video_features = self.get_model().mm_projector(frames_features)
        elif self.config.mm_projector_type == "spatial_pool":
            video_features = self.get_model().mm_projector(frames_features)
        elif "tc_connector" in self.config.mm_projector_type or "tp_connector" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features,
                                                           cls_inference = cls_inference,
                                                           cls_training = cls_training,
                                                           cls_demo = cls_demo, 
                                                           frames_features_shape = frames_features_shape)
        elif "mamba" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features,
                                                           cls_inference = cls_inference,
                                                           cls_training = cls_training,
                                                           cls_demo = cls_demo, 
                                                           frames_features_shape = frames_features_shape,
                                                           prompt_time_input_ids = prompt_time_input_ids,
                                                           prompt_time_lable = prompt_time_lable)
        else:
            raise Exception(f"Unsupported projector type {self.config.mm_projector_type}!!!")
        return video_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, X_modalities
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or X_modalities is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, past_key_values, None, labels

        Xs, keys = X_modalities
        X_features = self.encode_images_or_videos(Xs, keys)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_X_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)).sum() == 0:
                cur_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_X_idx += 1
                continue

            X_token_indices = torch.where(
                torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)
            )[0]

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            while X_token_indices.numel() > 0:
                cur_X_features = X_features[cur_X_idx]
                X_token_start = X_token_indices[0].item()

                if X_token_start > 0:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:X_token_start]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:X_token_start])

                cur_new_input_embeds.append(cur_X_features)
                if labels is not None:
                    cur_new_labels.append(
                        torch.full(
                            (cur_X_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                    cur_labels = cur_labels[X_token_start + 1 :]

                cur_X_idx += 1
                cur_input_ids = cur_input_ids[X_token_start + 1 :]
                X_token_indices = torch.where(
                    torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)
                )[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)

            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        bsz = len(new_input_embeds)
        new_lens = [x.shape[0] for x in new_input_embeds]
        max_len = max(new_lens)

        if any(x.shape[0] != max_len for x in new_input_embeds):
            padded = []
            for x in new_input_embeds:
                pad_len = max_len - x.shape[0]
                if pad_len > 0:
                    pad = torch.zeros((pad_len, x.shape[1]), dtype=x.dtype, device=x.device)
                    x = torch.cat([x, pad], dim=0)
                padded.append(x)
            new_input_embeds = torch.stack(padded, dim=0)
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)

        if labels is not None:
            padded_labels = []
            for y in new_labels:
                pad_len = max_len - y.shape[0]
                if pad_len > 0:
                    pad = torch.full((pad_len,), IGNORE_INDEX, dtype=y.dtype, device=y.device)
                    y = torch.cat([y, pad], dim=0)
                padded_labels.append(y)
            new_labels = torch.stack(padded_labels, dim=0)

        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=new_input_embeds.device)
        for i, L in enumerate(new_lens):
            attention_mask[i, :L] = 1

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def frame_sample(self, duration, mode='uniform', local_fps=None,num_frames = None):
        import numpy as np
        if mode == 'uniform':
            seg_size = float(duration - 1) / num_frames

            frame_ids = []
            for i in range(num_frames):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                frame_ids.append((start + end) // 2)

            return frame_ids

        elif mode == 'fps':
            segment_len = 2
            return np.arange(1, duration, segment_len, dtype=int)
        else:
            raise ImportError(f'Unsupported frame sampling mode: {mode}')
    
    def prepare_inputs_labels_for_multimodal_score(
         self, input_ids, attention_mask, past_key_values, labels, X_modalities,timestamp,half,**kwargs    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or X_modalities is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, past_key_values, None, labels

        Xs, keys = X_modalities
        X_features = self.encode_images_or_videos_score(Xs)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_X_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)).sum() == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_X_features = X_features[cur_X_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_X_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_X_idx += 1 
                continue

            X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0] 
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            
            while X_token_indices.numel() > 0:
                cur_X_features = X_features[cur_X_idx]
                X_token_start = X_token_indices[0]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:X_token_start])) 
                cur_new_input_embeds.append(cur_X_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:X_token_start])
                    cur_new_labels.append(torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[X_token_start+1:]

                cur_X_idx += 1
                cur_input_ids = cur_input_ids[X_token_start+1:] 
                X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
    def exponential_sampling(self, tensor, percentage=0.6):
        n = tensor.size(0)
        num_samples =  1 if int(percentage * n) == 0 else int(percentage * n)
        indices = torch.linspace(0, n - 1, num_samples).int().tolist() 
        return tensor[indices]

    def similarity_sampling(self, tensor, percentage=0.6):
        last_token = tensor[-1] 
        similarities = F.cosine_similarity(tensor, last_token.unsqueeze(0), dim=1)
        sorted_indices = torch.argsort(similarities, descending=True)
        top_k = max(int(percentage * len(sorted_indices)), 1)
        top_60_indices = sorted(sorted_indices[:top_k].tolist())
        return tensor[top_60_indices]

    def prepare_inputs_labels_for_multimodal_score_stream(
        self, input_ids, attention_mask, past_key_values, labels, X_modalities, timestamp,
        sample_per=0.5, sample_type="all", **kwargs
    ):
        self.train_iteration += 1

        model_type = kwargs.pop("model_type", None)
        data_type = kwargs.pop("data_type", None)

        if model_type == "cls":
            if data_type == "train":
                cls_inference = False
                cls_trainging = True
            else:
                cls_inference = True
                cls_trainging = False
        else:
            cls_inference = False
            cls_trainging = False

        Xs, keys = X_modalities
        exactor_output, feature_idx = self.encode_images_or_videos_score_cls_video_cls_autoregressive(
            Xs,
            cls_training=cls_trainging,
            cls_inference=cls_inference,
            prompt_time_input_ids=input_ids,
            prompt_time_lable=labels,
        )

        if cls_inference or cls_trainging:
            return None, None, None, None, None, exactor_output
        else:
            X_features = exactor_output

        start_feature_idx = [0] + feature_idx[:-1]

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_X_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)).sum() == 0:
                cur_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_X_idx += 1
                continue

            X_token_indices = torch.where(
                torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)
            )[0]

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            while X_token_indices.numel() > 0:
                if sample_type == "log":
                    cur_X = self.exponential_sampling(
                        X_features[0][start_feature_idx[cur_X_idx]:feature_idx[cur_X_idx]],
                        sample_per
                    )
                elif sample_type == "similarity":
                    cur_X = self.similarity_sampling(
                        X_features[0][start_feature_idx[cur_X_idx]:feature_idx[cur_X_idx]],
                        sample_per
                    )
                else:
                    cur_X = X_features[0][start_feature_idx[cur_X_idx]:feature_idx[cur_X_idx]]

                X_token_start = X_token_indices[0].item()

                if X_token_start > 0:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:X_token_start]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:X_token_start])

                cur_new_input_embeds.append(cur_X)
                if labels is not None:
                    cur_new_labels.append(
                        torch.full((cur_X.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                    )
                    cur_labels = cur_labels[X_token_start + 1 :]

                cur_X_idx += 1
                cur_input_ids = cur_input_ids[X_token_start + 1 :]
                X_token_indices = torch.where(
                    torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)
                )[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)

            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        bsz = len(new_input_embeds)
        new_lens = [x.shape[0] for x in new_input_embeds]
        max_len = max(new_lens)

        padded = []
        for x in new_input_embeds:
            pad_len = max_len - x.shape[0]
            if pad_len > 0:
                pad = torch.zeros((pad_len, x.shape[1]), dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], dim=0)
            padded.append(x)
        new_input_embeds = torch.stack(padded, dim=0)

        if labels is not None:
            padded_labels = []
            for y in new_labels:
                pad_len = max_len - y.shape[0]
                if pad_len > 0:
                    pad = torch.full((pad_len,), IGNORE_INDEX, dtype=y.dtype, device=y.device)
                    y = torch.cat([y, pad], dim=0)
                padded_labels.append(y)
            new_labels = torch.stack(padded_labels, dim=0)

        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=new_input_embeds.device)
        for i, L in enumerate(new_lens):
            attention_mask[i, :L] = 1

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, None

    def prepare_inputs_labels_for_multimodal_score_stream_inference(
         self, input_ids, attention_mask, past_key_values, labels, X_modalities, score_video=None, force_gate: bool=False, **kwargs):
        vision_tower = self.get_vision_tower()
        frames_features_shape = kwargs.get("frames_features_shape", None)
        if frames_features_shape is None:
            frames_features_shape = []

        if vision_tower is None or X_modalities is None or input_ids.shape[1] == 1:
            pred = 1 if force_gate else 0
            frames_features = kwargs.get("frames_features", None)
            return input_ids, attention_mask, past_key_values, None, labels, pred, frames_features

        Xs, keys = X_modalities

        X_features, cls_feature, frames_features = self.encode_images_or_videos_score_cls_inference_allframe(
        Xs,
        kwargs["frames_features"],
        frames_features_shape=frames_features_shape,
    )
        gate_mode = os.getenv("STREAMMIND_GATE_MODE", "gpu").strip().lower()
        if gate_mode == "cpu":
            if isinstance(X_features, torch.Tensor) and X_features.device.type != "cpu":
                X_features = X_features.detach().to("cpu").contiguous().pin_memory()
        else:
            if isinstance(X_features, torch.Tensor):
                X_features = X_features.detach()
        if cls_feature is None:
            pred = 0
        else:
            if cls_feature.dim() == 2:
                gate_logits = cls_feature[-1]        
            else:
                gate_logits = cls_feature            

            if gate_logits.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                gate_logits = gate_logits.float()

            cls_prob = torch.softmax(gate_logits, dim=-1)
            pred = int(torch.argmax(cls_prob, dim=-1).item())

        if force_gate or os.environ.get("STREAMMIND_FORCE_GATE", "0") == "1" or os.environ.get("STREAMMIND_FORCE_SPEAK", "0") == "1":
            pred = 1

        if pred == 0:
            return input_ids, attention_mask, past_key_values, None, labels, 0, frames_features
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_X_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)).sum() == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_X_features = X_features[cur_X_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_X_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_X_idx += 1 
                continue

            X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0] 
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            
            while X_token_indices.numel() > 0:
                cur_X_features = X_features[cur_X_idx]

            K_req = int(os.environ.get("STREAMMIND_K", "64"))
            avail_K = int(cur_X_features.shape[0])
            K_eff = min(max(K_req, 0), avail_K)

            self._dbg_avail_K = avail_K
            self._dbg_actual_K = K_eff

            cur_X_cpu = cur_X_features[:K_eff]

            t0 = time.perf_counter()
            cur_X_features = cur_X_cpu.to(device=cur_input_ids.device, non_blocking=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            self._dbg_xfer_k_ms = float(1000.0 * (t1 - t0))
            self._dbg_xfer_k_bytes = int(cur_X_cpu.numel() * cur_X_cpu.element_size())

            X_token_start = X_token_indices[0]

            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:X_token_start]))
            if kwargs.get("past_review_caption", None) is not None:
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(kwargs["past_review_caption"][0].to(cur_input_ids.device))
                )
            cur_new_input_embeds.append(cur_X_features)

            if labels is not None:
                cur_new_labels.append(cur_labels[:X_token_start])
                cur_new_labels.append(torch.full(
                    (cur_X_features.shape[0],), IGNORE_INDEX,
                    device=cur_labels.device, dtype=cur_labels.dtype
                ))
                if kwargs.get("past_review_caption", None) is not None:
                    cap_len = self.get_model().embed_tokens(kwargs["past_review_caption"][0]).shape[0]
                    cur_new_labels.append(torch.full(
                        (cap_len,), IGNORE_INDEX,
                        device=cur_labels.device, dtype=cur_labels.dtype
                    ))
                cur_labels = cur_labels[X_token_start + 1:]

            cur_X_idx += 1
            cur_input_ids = cur_input_ids[X_token_start + 1:]
            X_token_indices = torch.where(torch.any(torch.stack(
                [cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]
            ), dim=0))[0]


            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            
            fixed = []
            H = self.get_model().embed_tokens.weight.shape[1] 

            for t in cur_new_input_embeds:
                if not torch.is_tensor(t):
                    t = torch.tensor(t, device=self.device)

                if t.dim() == 0:
                    t = t.long()
                    t = self.get_model().embed_tokens(t.view(1).to(self.device)) 

                elif t.dim() == 1:
                    if t.numel() == H and t.dtype.is_floating_point:
                        t = t.unsqueeze(0).to(self.device)
                    else:
                        t = t.long()
                        t = self.get_model().embed_tokens(t.to(self.device)) 
                else:
                    t = t.to(self.device)

                fixed.append(t)

            cur_new_input_embeds = torch.cat(fixed, dim=0)

            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
              
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
            else:
                attention_mask = torch.full((new_input_embeds.shape[0],new_input_embeds.shape[1]), True, device=new_input_embeds.device)

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, pred, frames_features

    def prepare_inputs_labels_for_multimodal_score_stream_inference_demo(
         self, input_ids, attention_mask, past_key_values, labels, X_modalities,**kwargs    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or X_modalities is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, past_key_values, None, labels

        Xs, keys = X_modalities
    
        X_features, cls_feature ,frames_features,interval_id = self.encode_images_or_videos_score_cls_inference_allframe_demo(
            Xs,kwargs["frames_features"], frames_features_shape = kwargs["interval_id_list"])

        softmax = nn.Softmax()
        cls_feature = softmax(cls_feature)
        pred = cls_feature.argmax(dim=0).item()
        if pred == 0:
            return None, None, None, None, None, pred, frames_features, interval_id
        else:
            kwargs["interval_id_list"].append(interval_id)
            start_feature_idx=[0]+kwargs["interval_id_list"][:-1]
            
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_X_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
        
            X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0] 
            cur_new_input_embeds = []
           
            while X_token_indices.numel() > 0:
                cur_X_features = X_features[0][start_feature_idx[cur_X_idx]:kwargs["interval_id_list"][cur_X_idx]]

                X_token_start = X_token_indices[0]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:X_token_start]))
                cur_new_input_embeds.append(cur_X_features)
                
                cur_X_idx += 1
                cur_input_ids = cur_input_ids[X_token_start+1:]
                X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)

        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        attention_mask = torch.full((new_input_embeds.shape[0],new_input_embeds.shape[1]), 1, device=new_input_embeds.device)

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, pred, frames_features, interval_id

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings  = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg  = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:]  = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def initialize_MM_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            for modal in ['IMAGE', 'VIDEO', 'AUDIO']:
                tokenizer.add_tokens([DEFAULT_MMODAL_PATCH_TOKEN[modal.upper()]], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = 0
            for modal in ['IMAGE', 'VIDEO', 'AUDIO']:
                num_new_tokens += tokenizer.add_tokens([DEFAULT_MMODAL_START_TOKEN[modal.upper()], DEFAULT_MMODAL_END_TOKEN[modal.upper()]], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 6
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False