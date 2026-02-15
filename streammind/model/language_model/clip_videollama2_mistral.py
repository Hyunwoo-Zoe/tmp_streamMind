
import os
from typing import List, Optional, Tuple, Union
import time
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    MistralConfig,
    MistralModel,
    MistralForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..videollama2_arch import Videollama2MetaModel, Videollama2MetaForCausalLM

class Videollama2MistralConfig(MistralConfig):
    model_type = "videollama2_mistral"

class Videollama2MistralModel(Videollama2MetaModel, MistralModel):
    config_class = Videollama2MistralConfig

    def __init__(self, config: MistralConfig):
        super(Videollama2MistralModel, self).__init__(config)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0].fill_(alpha)
            self.alpha[1:].fill_(1 - alpha)

        self.gamma = gamma

        print("Focal Loss:")
        print("    Alpha = {}".format(self.alpha))
        print("    Gamma = {}".format(self.gamma))

    def forward(self, preds, labels):
        # 여기 아래 줄부터 모두 한 단계 들여쓰기가 되어야 합니다.
        preds = preds.view(-1, preds.size(-1)).float() 
        labels_view = labels.view(-1, 1).long()
        
        alpha = self.alpha.to(device=preds.device, dtype=preds.dtype)

        log_pt = F.log_softmax(preds, dim=1)
        pt = torch.exp(log_pt)

        log_pt = log_pt.gather(1, labels_view)
        pt = pt.gather(1, labels_view)

        loss = -torch.pow((1 - pt), self.gamma) * log_pt

        label_flatten = labels_view.view(-1)
        alpha_w = alpha.gather(0, label_flatten).view(-1, 1)
        loss = alpha_w * loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
class Videollama2MistralForCausalLM(MistralForCausalLM, Videollama2MetaForCausalLM):
    config_class = Videollama2MistralConfig

    def __init__(self, config, **kwargs):
        super(MistralForCausalLM, self).__init__(config)

        self.train_iteration = 0
        self.model = Videollama2MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

        # [핵심 1] 스트리밍 상태(Memory) 변수들
        # self.frame_feature: Mamba/SSM이 관리하는 '비디오 문맥 압축본' (계속 갱신됨)

        self.frame_feature = None
        self.past_review_caption = None
        self.past_review_caption_list = []
        self.interval_id_list = []
        self.loss_fct = CrossEntropyLoss()
        self.time_list = []

        self.sample_per = kwargs.pop("sample_per", 0.5)
        self.sample_type = kwargs.pop("sample_type", "ssss")
        self._gate_prev = 0            
        self._last_spoken_text = None   

    def get_model(self):
        return self.model

    def _compute_past_len(self, past_key_values, attention_mask, cur_len: int) -> int:
        if past_key_values is None:
            return 0

        if hasattr(past_key_values, "get_seq_length"):
            try:
                pl = past_key_values.get_seq_length()
                if pl is None:
                    pl = 0
                pl = int(pl)
                if pl < 0:
                    pl = 0
                return pl
            except Exception:
                pass

        try:
            k = past_key_values[0][0]
            if k.dim() == 4:
                pl = int(k.shape[2])
            else:
                pl = int(k.shape[-2])
            if pl < 0:
                pl = 0
            return pl
        except Exception:
            pass

        if attention_mask is not None:
            try:
                pl = int(attention_mask.shape[1] - cur_len)
                if pl < 0:
                    pl = 0
                return pl
            except Exception:
                pass

        return 0

    def _force_attention_mask(self, attention_mask, bsz: int, total_len: int, device) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones((bsz, total_len), device=device, dtype=torch.long)

        if attention_mask.dim() != 2:
            attention_mask = attention_mask.view(bsz, -1)

        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.to(dtype=torch.long)
        elif attention_mask.dtype not in (torch.long, torch.int64, torch.int32):
            attention_mask = attention_mask.to(dtype=torch.long)

        cur_am_len = attention_mask.shape[1]
        if cur_am_len == total_len:
            return attention_mask

        if cur_am_len > total_len:
            return attention_mask[:, :total_len]

        pad = torch.ones((bsz, total_len - cur_am_len), device=device, dtype=attention_mask.dtype)
        return torch.cat([attention_mask, pad], dim=1)

    def _build_position_ids(self, bsz: int, past_len: int, cur_len: int, device) -> torch.LongTensor:
        position_ids = torch.arange(past_len, past_len + cur_len, device=device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        max_pos = getattr(self.config, "max_position_embeddings", None)
        if max_pos is not None:
            max_pos = int(max_pos)
            if max_pos > 0:
                position_ids = position_ids.clamp(min=0, max=max_pos - 1)

        return position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cls_output=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None and past_key_values is None:
            if "timestamp" in kwargs.keys():
                (
                    input_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    cls_output,
                ) = self.prepare_inputs_labels_for_multimodal_score_stream(
                    input_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    sample_per=self.sample_per,
                    sample_type=self.sample_type,
                    **kwargs,
                )
            else:
                (
                    input_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                )

            if cls_output is not None:
                return cls_output

            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("Both inputs_embeds and input_ids are None.")
                inputs_embeds = self.get_model().embed_tokens(input_ids)

            bsz = inputs_embeds.shape[0]
            cur_len = inputs_embeds.shape[1]
            device = inputs_embeds.device

            past_len = self._compute_past_len(past_key_values, attention_mask, cur_len)

            total_len = past_len + cur_len
            attention_mask = self._force_attention_mask(attention_mask, bsz, total_len, device)

            position_ids = self._build_position_ids(bsz, past_len, cur_len, device)

            if position_ids.dtype != torch.long:
                position_ids = position_ids.to(dtype=torch.long)
            if position_ids.min().item() < 0:
                position_ids = position_ids.clamp_min(0)

            llm_output = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                position_ids=position_ids,
                **kwargs,
            )

            llm_eval = kwargs.pop("llm_eval", None)
            if llm_eval:
                return llm_output, labels

            return llm_output

    @torch.no_grad()
    def generate(self, inputs=None, images_or_videos=None, modal_list=None, **kwargs):
        inputs = kwargs.pop("input_ids", inputs)
        kwargs.pop("input_ids", None)
        kwargs.pop("position_ids", None)

        attention_mask = kwargs.pop("attention_mask", None)
        kwargs.pop("score_video", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported here (use inputs).")

        if inputs is None:
            raise ValueError("generate: inputs (input_ids) is None")

        if images_or_videos is not None:
            (
                _input_ids,
                attention_mask,
                _past_key_values,
                inputs_embeds,
                _labels,
                _cls,
            ) = self.prepare_inputs_labels_for_multimodal_score_stream(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                timestamp=None,
                model_type="llm",
                X_modalities=[images_or_videos, modal_list],
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if attention_mask is None:
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                device=inputs_embeds.device,
                dtype=torch.long,
            )
        else:
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.to(dtype=torch.long)

            if attention_mask.dim() != 2:
                attention_mask = attention_mask.view(inputs_embeds.shape[0], -1)

            if attention_mask.shape[1] != inputs_embeds.shape[1]:
                bsz = inputs_embeds.shape[0]
                total_len = inputs_embeds.shape[1]
                attention_mask = torch.ones((bsz, total_len), device=inputs_embeds.device, dtype=torch.long)

        return super().generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    @torch.no_grad()
    def stream_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images_or_videos: Optional[torch.Tensor] = None,
        modal_list: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # 인자 정리
        kwargs.pop("position_ids", None)
        kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        score_video = kwargs.pop("score_video", None)
        tokenizer = kwargs.pop("tokenizer", None)
        return_cls = kwargs.pop("return_cls", False)
        only_cls   = kwargs.pop("only_cls", False)
        return_metrics = kwargs.pop("return_metrics", False) 
        bench_gate_only = kwargs.pop("bench_gate_only", False) # [추가] 벤치마크 플래그

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # 과거 캡션 처리
        if self.past_review_caption is not None and tokenizer is not None:
            past_caption_ids = tokenizer(
                self.past_review_caption,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids
        else:
            past_caption_ids = None

        t_e2e0 = time.perf_counter()
        gate_mode = os.getenv("STREAMMIND_GATE_MODE", "gpu")

        def _dbg_fields():
            return {
                "gate_mode": getattr(self, "_dbg_gate_mode", gate_mode),
                "clip_ms": getattr(self, "_dbg_clip_ms", 0.0),
                "offload_ms": getattr(self, "_dbg_offload_ms", 0.0),
                "xfer_gate_ms": getattr(self, "_dbg_xfer_gate_ms", 0.0),
                "gate_compute_ms": getattr(self, "_dbg_gate_compute_ms", 0.0),
                
                # [추가됨] 세부 브레이크다운
                "mamba_ms": getattr(self, "_dbg_mamba_ms", 0.0),
                "trans_ms": getattr(self, "_dbg_trans_ms", 0.0),
                
                "xfer_k_ms": getattr(self, "_dbg_xfer_k_ms", 0.0),
                "avail_K": getattr(self, "_dbg_avail_K", 0),
                "actual_K": getattr(self, "_dbg_actual_K", 0),
            }

        def _fps_cap(wall_ms: float):
            return (1000.0 / wall_ms) if (wall_ms is not None and wall_ms > 0) else None

        disable_debounce = os.getenv("STREAMMIND_DISABLE_DEBOUNCE", "0") == "1"
        disable_text_dedup = os.getenv("STREAMMIND_DISABLE_TEXT_DEDUP", "0") == "1"
        requested_K = int(os.getenv("STREAMMIND_K", "0") or "0")

        # ----------------------------------------------------------------------
        # 1. Vision Encoding & Gate 실행 (arch.py 내부 로직 실행)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            _,
            cls_pred,
            self.frame_feature,
        ) = self.prepare_inputs_labels_for_multimodal_score_stream_inference(
            input_ids=inputs,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=None,
            X_modalities=[images_or_videos, ["video"]],
            past_review_caption=past_caption_ids,
            frames_features=self.frame_feature,
            frames_features_shape=self.interval_id_list,
            score_video=score_video,
            force_gate=False,
        )

        # ----------------------------------------------------------------------
        # 2. [핵심] 벤치마크 모드일 경우 여기서 즉시 리턴 (위치 변경됨)
        # ----------------------------------------------------------------------
        # LLM 생성을 건너뛰고 측정된 시간(metrics)만 반환합니다.
        if bench_gate_only:
            wall_ms = 1000.0 * (time.perf_counter() - t_e2e0)
            metrics = {
                "cls_pred": int(cls_pred) if cls_pred is not None else 0,
                "requested_K": requested_K,
                "gate_ms": 1000.0 * (time.perf_counter() - t0),
                "decode_gpu_ms": 0.0,
                "num_new_tokens": 0,
                "wall_ms": wall_ms,
                "fps_cap": _fps_cap(wall_ms),
                **_dbg_fields(), # 여기서 clip_ms, gate_compute_ms 등이 들어감
            }
            if return_metrics:
                return (None, cls_pred, metrics)
            return (None, cls_pred)

        # ----------------------------------------------------------------------
        # 3. 기존 LLM 생성 로직 (벤치마크 아닐 때만 실행)
        # ----------------------------------------------------------------------
        if inputs_embeds is None:
            self._gate_prev = 0
            wall_ms = 1000.0 * (time.perf_counter() - t_e2e0)
            metrics = {
                "cls_pred": 0,
                "requested_K": requested_K,
                "gate_ms": 1000.0 * (time.perf_counter() - t0),
                "decode_gpu_ms": 0.0,
                "num_new_tokens": 0,
                "wall_ms": wall_ms,
                "fps_cap": _fps_cap(wall_ms),
                **_dbg_fields(),
            }
            if return_cls and return_metrics:
                return (None, 0, metrics)
            if return_cls:
                return (None, 0)
            return None if not return_metrics else (None, metrics)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        gate_ms = 1000.0 * (t1 - t0)
        # (로그 출력 부분 생략 가능) ...

        prev = getattr(self, "_gate_prev", 0)

        # Gate가 0(대답 안 함)이면 스킵
        if cls_pred == 0:
            self._gate_prev = 0
            wall_ms = 1000.0 * (time.perf_counter() - t_e2e0)
            metrics = {
                "cls_pred": 0,
                "requested_K": requested_K,
                "gate_ms": gate_ms,
                "decode_gpu_ms": 0.0,
                "num_new_tokens": 0,
                "wall_ms": wall_ms,
                "fps_cap": _fps_cap(wall_ms),
                **_dbg_fields(),
            }
            if return_cls and return_metrics:
                return (None, 0, metrics)
            if return_cls:
                return (None, 0)
            return None if not return_metrics else (None, metrics)

        if only_cls:
            self._gate_prev = 1
            wall_ms = 1000.0 * (time.perf_counter() - t_e2e0)
            metrics = {
                "cls_pred": 1,
                "requested_K": requested_K,
                "gate_ms": gate_ms,
                "decode_gpu_ms": 0.0,
                "num_new_tokens": 0,
                "wall_ms": wall_ms,
                "fps_cap": _fps_cap(wall_ms),
                **_dbg_fields(),
            }
            if return_cls and return_metrics:
                return (None, 1, metrics)
            if return_cls:
                return (None, 1)
            return None if not return_metrics else (None, metrics)
        
        # Debounce 로직
        if prev == 1 and not disable_debounce:
            self._gate_prev = 1
            wall_ms = 1000.0 * (time.perf_counter() - t_e2e0)
            metrics = {
                "cls_pred": 1,
                "requested_K": requested_K,
                "gate_ms": gate_ms,
                "decode_gpu_ms": 0.0,
                "num_new_tokens": 0,
                "wall_ms": wall_ms,
                "fps_cap": _fps_cap(wall_ms),
                **_dbg_fields(),
            }
            if return_cls and return_metrics:
                return (None, 1, metrics)
            if return_cls:
                return (None, 1)
            return None if not return_metrics else (None, metrics)

        self._gate_prev = 1

        if attention_mask is None:
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                device=inputs_embeds.device,
                dtype=torch.long,
            )
        elif attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.to(dtype=torch.long)

        # ----------------------------------------------------------------------
        # 4. LLM 생성 (Decording)
        # ----------------------------------------------------------------------
        decode_gpu_ms = 0.0
        if torch.cuda.is_available():
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            ev0.record()

        output = super().generate(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if torch.cuda.is_available():
            ev1.record()
            torch.cuda.synchronize()
            decode_gpu_ms = float(ev0.elapsed_time(ev1))

        if tokenizer is None:
            return output

        prompt_len = int(inputs_embeds.shape[1])
        seq_len = int(output.shape[1])
        num_new_tokens = max(0, seq_len - prompt_len)

        text = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()

        # Text Dedup 로직
        last = getattr(self, "_last_spoken_text", None)
        if (not disable_text_dedup) and (last == text):
            metrics = {
                "cls_pred": 1,
                "requested_K": requested_K,
                "actual_K": getattr(self, "_dbg_actual_K", None),
                "xfer_k_bytes": getattr(self, "_dbg_xfer_k_bytes", None),
                "gate_ms": gate_ms,
                "decode_gpu_ms": decode_gpu_ms,
                "num_new_tokens": num_new_tokens,
                "wall_ms": 1000.0 * (time.perf_counter() - t_e2e0),
                **_dbg_fields(),
            }
            if return_cls and return_metrics:
                return (None, 1, metrics)
            return None if not return_metrics else (None, metrics)

        self._last_spoken_text = text

        if self.past_review_caption is None:
            self.past_review_caption = text
        else:
            self.past_review_caption += text

        metrics = {
            "cls_pred": int(cls_pred),
            "requested_K": requested_K,
            "gate_ms": gate_ms,
            "decode_gpu_ms": decode_gpu_ms,
            "num_new_tokens": num_new_tokens,
            "wall_ms": (wall_ms := 1000.0 * (time.perf_counter() - t_e2e0)),
            "fps_cap": _fps_cap(wall_ms),
            **_dbg_fields(), # 마지막 결과 리턴 시에도 metrics 포함
        }

        if return_cls and return_metrics:
            return (text, int(cls_pred), metrics)
        if return_cls:
            return (text, int(cls_pred))
        return text if not return_metrics else (text, metrics)

    @torch.no_grad()
    def stream_generate_demo(
        self,
        inputs: Optional[torch.Tensor] = None,
        images_or_videos: Optional[torch.Tensor] = None,
        modal_list: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        kwargs.pop("position_ids", None)
        kwargs.pop("input_ids", None)

        attention_mask = kwargs.pop("attention_mask", None)
        kwargs.pop("score_video", None)
        tokenizer = kwargs.pop("tokenizer", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            _,
            cls_pred,
            self.frame_feature,
            interval_id,
        ) = self.prepare_inputs_labels_for_multimodal_score_stream_inference_demo(
            input_ids=inputs,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=None,
            X_modalities=[images_or_videos, ["video"]],
            frames_features=self.frame_feature,
            interval_id_list=self.interval_id_list,
        )

        if cls_pred == 0 and (not force_gate):
            return None, cls_pred

        if attention_mask is None:
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                device=inputs_embeds.device,
                dtype=torch.long,
            )
        elif attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.to(dtype=torch.long)

        output_idx = super().generate(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if tokenizer is None:
            return output_idx, cls_pred

        text = tokenizer.batch_decode(output_idx, skip_special_tokens=True)[0].strip()
        return text, cls_pred

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs["images"] = images
        return _inputs

AutoConfig.register("videollama2_mistral", Videollama2MistralConfig)
AutoModelForCausalLM.register(Videollama2MistralConfig, Videollama2MistralForCausalLM)
