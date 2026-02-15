
import os
import re
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.regnet import RegStage
from timm.models.layers import LayerNorm, LayerNorm2d, trunc_normal_
from transformers import TRANSFORMERS_CACHE
from dataclasses import dataclass
from .ssm import VideoMamba
from pytorch_lightning import LightningModule
from streammind.constants import IGNORE_INDEX
import inspect
import math

@dataclass
class SSMConfig:
    d_code = 1024
    d_model = 2048
    n_ssm = 1
    n_classes = 400
    lr = 1.4e-4
    lr_min = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0.02
    scheduler = "plateau"


def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read().strip()
    folder = os.path.join(repo_cache, "snapshots", revision)

    return folder

def load_mm_projector(model_path, cache_dir=None, token=None):
    if os.path.exists(os.path.join(model_path, "mm_projector.bin")):
        is_local = True
        folder = model_path
    else:
        is_local = False
        folder = parse_snapshot_folder(
            model_path, cache_dir=cache_dir, repo_type="model"
        )
        if not os.path.exists(os.path.join(folder, "mm_projector.bin")):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_path, cache_dir=cache_dir, token=token)

    mm_projector_weights = torch.load(os.path.join(folder, "mm_projector.bin"), map_location="cpu")
    
    mm_projector_weights = {
        k: v.to(torch.bfloat16) for k, v in mm_projector_weights.items()
    }
    return mm_projector_weights

class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}

class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def count_parameters(model):
    value =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    return value * 4 / 1024 / 1024

def build_vision_projector(config, delay_load=False, **kwargs):
    #projector_type = getattr(config, "mm_projector_type", "linear")
    projector_type = "mamba"
    print(f"[INFO] Forcing projector_type to: {projector_type}")
    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == "mamba":
        projector_model = Video_Mamba_seq(config)
        print(f"Trainable parameters: {count_parameters(projector_model)}MB")

        return projector_model 
    elif projector_type == "stc_connector":
        projector_model = STCConnector(config)
        print(f"Trainable parameters: {count_parameters(projector_model)}MB")
        return projector_model
    elif projector_type == "stp_connector":
        return STPConnector(config)
    elif projector_type == "stc_connector_v35":
        return STCConnectorV35(config)
    elif projector_type == "spatial_conv":
        return SpatialConv(config)
    elif projector_type == "spatial_pool":
        return SpatialPool(config)
    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")

class PreNet(nn.Module):
    def __init__(self, d_code, d_model):
        super(PreNet, self).__init__()
        self.fc3 = nn.Linear(d_code, d_model)

    def forward(self, x):
        x = self.fc3(x)
        x = F.leaky_relu(x)
        return x

class PostNet(nn.Module):
    def __init__(self, d_model, n_class):
        super(PostNet, self).__init__()
        self.fc3 = nn.Linear(d_model, n_class)

    def forward(self, x):
        x = F.leaky_relu(x)
        x = self.fc3(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.,norm_layer=LayerNorm):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(dim, int(expansion_factor * dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(expansion_factor * dim), dim))
        self.ln = norm_layer(dim)
    def forward(self, x):
        return x + self.fn(self.ln(x))

class TextProj(nn.Module):
    def __init__(self, embedding_dim=4096, output_dim=512, norm_layer=LayerNorm):
        super().__init__()
        self.embedding_dim = embedding_dim
        expansion_factor = 2
        dropout = 0
        proj_bias = True
        num_layers_text = 4
        self.text_adaptor = nn.Sequential(
            *[LinearBlock(embedding_dim, expansion_factor, dropout) for _ in range(num_layers_text)],
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, output_dim, bias=proj_bias),
            )
        self.grad_checkpointing = False
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def lock(self, unlocked_layers, freeze_layer_norm):
        for param in self.text_adaptor.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, text, return_all_features: bool=False):
        x = self.text_adaptor(text)
        return x

from transformers import MistralConfig, MistralForCausalLM,Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class MistralForCausalLM_cls(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        import inspect

        model_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sig = inspect.signature(self.model.forward)
        if "cache_position" in sig.parameters:
            model_kwargs["cache_position"] = cache_position

        outputs = self.model(**model_kwargs)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            weight_list = [1]* (self.config.vocab_size-2)
            weight_list.append(0.15)
            weight_list.append(0.85)
            loss_fct = CrossEntropyLoss(weight=torch.tensor(weight_list).to(shift_logits.device))
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ClsNet(nn.Module):
    def __init__(self, d_model, depth, n_class ):
        super(ClsNet, self).__init__()
        mis_config = MistralConfig()
        mis_config.vocab_size = 2
        mis_config.num_hidden_layers = 4
        #self.cls_model = MistralForCausalLM_cls(config=mis_config).to(torch.bfloat16)
        self.cls_model = MistralForCausalLM_cls(config=mis_config)

    def forward(self, x, cls_labels, cls_attention_mask):
        # --- FIX: align dtype/device with cls_model ---
        p = next(self.cls_model.parameters())
        model_device = p.device
        model_dtype = p.dtype

        if x.device != model_device:
            x = x.to(device=model_device)
        if x.dtype != model_dtype:
            x = x.to(dtype=model_dtype)

        if cls_labels is not None:
            if cls_labels.device != model_device:
                cls_labels = cls_labels.to(device=model_device)
            if cls_labels.dtype not in (torch.long, torch.int64, torch.int32):
                cls_labels = cls_labels.long()

        if cls_attention_mask is not None:
            if cls_attention_mask.device != model_device:
                cls_attention_mask = cls_attention_mask.to(device=model_device)
            if cls_attention_mask.dim() == 4:
                cls_attention_mask = cls_attention_mask[:, 0, 0, :]
            elif cls_attention_mask.dim() == 3:
                cls_attention_mask = cls_attention_mask[:, 0, :]
            if cls_attention_mask.dtype not in (torch.bool, torch.long, torch.int64, torch.int32):
                cls_attention_mask = cls_attention_mask.to(torch.long)
        # --- FIX END ---

        x = self.cls_model(
            inputs_embeds=x,
            labels=cls_labels,
            attention_mask=cls_attention_mask
        )
        return x

import time
import os

class Video_Mamba_seq(LightningModule):
    def __init__(self, model_config):
        super(Video_Mamba_seq, self).__init__()
        self.pre_net = PreNet(model_config.mm_hidden_size, model_config.hidden_size)
        mamba_config = SSMConfig()
        mamba_config.d_code=model_config.hidden_size
        mamba_config.d_model=model_config.hidden_size
        self.mamba_model = VideoMamba(mamba_config)
        self.post_net = PostNet(model_config.hidden_size, model_config.hidden_size)
        self.cls_net = ClsNet( d_model=model_config.hidden_size, depth=4, n_class=2)
        self.time_list = []
        self.videoid = 0 

    def step(self, x, state=None, frames_features_shape=None):
        """
        Mamba 아키텍처용 O(1) 증분 처리
        """
        # [타이머 시작] Mamba 연산
        if x.is_cuda: torch.cuda.synchronize()
        t_mamba_start = time.perf_counter()

        b, t, n, h = x.shape
        # Spatial Pooling: [B, 1, N, H] -> [B, 1, H]
        x_t = torch.mean(x, dim=2) 
        
        # 상태 초기화
        if state is None:
            state = []
            for ssm in self.mamba_model.ssms:
                # Mamba 기본 설정: d_conv=4, d_state=16
                d_inner = ssm.d_inner
                d_conv = ssm.d_conv
                d_state = ssm.d_state
                c_s = torch.zeros(b, d_inner, d_conv, device=x.device, dtype=x.dtype)
                s_s = torch.zeros(b, d_inner, d_state, device=x.device, dtype=x.dtype)
                state.append((c_s, s_s))

        # 1. Pre-net
        hidden = self.pre_net(x_t) # [B, 1, D]

        # 2. Mamba Layers Loop
        new_state = []
        for i, ssm in enumerate(self.mamba_model.ssms):
            prev_c_s, prev_s_s = state[i]
            
            # ssm.step 호출
            out_step, next_c_s, next_s_s = ssm.step(hidden, prev_c_s, prev_s_s)
            
            # Residual Connection
            hidden = hidden + out_step
            hidden = self.mamba_model.norm_fn(hidden)
            
            new_state.append((next_c_s, next_s_s))

        # 3. Post-net
        x_out = self.post_net(hidden)

        # [타이머 종료] Mamba 시간
        if x.is_cuda: torch.cuda.synchronize()
        self.last_mamba_ms = (time.perf_counter() - t_mamba_start) * 1000

        # [타이머 시작] Gate 연산
        t_gate_start = time.perf_counter()

        # 4. Gate (ClsNet)
        eos_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([0], device=x.device)).view(1, 1, -1)
        input_embed = torch.cat([x_out, eos_target.expand(b, -1, -1)], dim=1)
        
        cls_output = self.cls_net(input_embed, cls_labels=None, cls_attention_mask=None)
        cls_last = cls_output.logits[:, -1, :]

        # [타이머 종료] Gate 시간
        if x.is_cuda: torch.cuda.synchronize()
        self.last_gate_ms = (time.perf_counter() - t_gate_start) * 1000

        return x_out, cls_last, new_state

    def forward(self, x, cls_inference=False, cls_training=False, cls_demo=False, frames_features_shape=[], prompt_time_input_ids=None, prompt_time_lable=None):
        # 0. 시간 저장 변수 초기화
        self.last_mamba_ms = 0.0 # Mamba/SSM 연산 시간
        self.last_gate_ms = 0.0  # Gate(Mistral) 연산 시간
        
        b, t, l, d = x.shape

        if os.getenv("STREAMMIND_DEBUG_MAMBA", "0") == "1" and not getattr(self, "_dbg_mamba_once", False):
            self._dbg_mamba_once = True
            print(f"[DBG_MAMBA] input x shape={tuple(x.shape)} dev={x.device} dtype={x.dtype}")
            print(f"[DBG_MAMBA] pre_net dtype={next(self.pre_net.parameters()).dtype} dev={next(self.pre_net.parameters()).device}")
            # VideoMamba 내부가 어떤 구현인지(대충이라도) 확인
            names = []
            for m in self.mamba_model.modules():
                n = type(m).__name__
                if "Mamba" in n or "SSM" in n or "selective" in n or "Pure" in n:
                    names.append(n)
            print(f"[DBG_MAMBA] impl_hits={sorted(set(names))[:20]}")


        # print(f"[RUNTIME] Mamba Sequence Length (T): {t}, Original Batch (B): {b}")

        # -------------------------------------------------------
        # 1. Mamba (SSM) 연산 구간 시작
        # -------------------------------------------------------
        if x.is_cuda: torch.cuda.synchronize()
        t_ssm_start = time.perf_counter()

        x = torch.mean(x, dim=2) 
        x = einops.rearrange(x, "b t d -> (b t) d", b=b, t=t)
        x = self.pre_net(x)
        x = einops.rearrange(x, "(b t) d -> b t d", b=b, t=t)
        x = self.mamba_model(x)
        x = einops.rearrange(x, "b t d -> (b t) d")
        x = self.post_net(x)
        x = einops.rearrange(x, "(b t) d -> b t d", b=b, t=t)
        
        if x.is_cuda: torch.cuda.synchronize()
        self.last_mamba_ms = (time.perf_counter() - t_ssm_start) * 1000
        # -------------------------------------------------------

        if cls_training or cls_inference:
            # --- 경로 A: Prompt 기반 복합 입력 (프롬프트 템플릿 사용) ---
            if prompt_time_input_ids is not None and prompt_time_input_ids.numel() > 1:
                # [데이터 전처리 부분 시작]
                pad_token_id = 0
                input_embeds = []
                cls_labels = []
                X_prompt_idx = (prompt_time_input_ids == -201).nonzero(as_tuple=False)[0, 1].item()
                X_pred_idx   = (prompt_time_lable == 32000).nonzero(as_tuple=False)[0, 1].item()
                prompt_template_inputs = self.cls_net.cls_model.model.embed_tokens(prompt_time_input_ids[:, :X_prompt_idx])
                prompt_template_inputs_requirements = self.cls_net.cls_model.model.embed_tokens(prompt_time_input_ids[:, X_prompt_idx + 1 : X_pred_idx])
                prompt_template_inputs_rest = self.cls_net.cls_model.model.embed_tokens(prompt_time_input_ids[:, X_pred_idx + 1 :])

                prompt_template_labels = prompt_time_lable[:, :X_prompt_idx].to(x.device)
                prompt_template_labels_requirements = prompt_time_lable[:, X_prompt_idx + 1 : X_pred_idx].to(x.device)
                prompt_template_labels_rest = prompt_time_lable[:, X_pred_idx + 1 :].to(x.device)

                eos_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([0]).to(x.device)).unsqueeze(0)
                caption_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([1]).to(x.device)).unsqueeze(0)
                
                start_feature_idx = [0] + frames_features_shape[:-1]
                for idx, end_frame_idx in enumerate(frames_features_shape):
                    cur_frame_feature = x[0][start_feature_idx[idx] : end_frame_idx]
                    if cur_frame_feature.shape[0] == 0: continue
                    if cur_frame_feature.shape[0] > 1:
                        input_embed = torch.cat([torch.cat([prompt_template_inputs, frame.unsqueeze(0).unsqueeze(0), prompt_template_inputs_requirements, eos_target, prompt_template_inputs_rest], dim=1) for frame in cur_frame_feature[:-1]])
                        cls_label = torch.cat([torch.cat([prompt_template_labels, torch.full((1,1),IGNORE_INDEX).to(x.device), prompt_template_labels_requirements, torch.tensor([[32000]]).to(x.device), prompt_template_labels_rest], dim=1) for _ in cur_frame_feature[:-1]])
                        input_embeds.append(input_embed)
                        cls_labels.append(cls_label)
                    input_embeds.append(torch.cat([prompt_template_inputs, cur_frame_feature[-1].unsqueeze(0).unsqueeze(0), prompt_template_inputs_requirements, caption_target, prompt_template_inputs_rest], dim=1))
                    cls_labels.append(torch.cat([prompt_template_labels, torch.full((1,1),IGNORE_INDEX).to(x.device), prompt_template_labels_requirements, torch.tensor([[32001]]).to(x.device), prompt_template_labels_rest], dim = 1))

                input_embed = torch.cat(input_embeds)
                cls_label = torch.cat(cls_labels)
                cls_label = cls_label[:4000]
                input_embed = input_embed[:4000]
                cls_attention_mask = (input_embed.abs().sum(dim=-1) != 0).to(dtype=torch.long)
                cls_label = cls_label.clone()
                cls_label[cls_label == 32000] = 0
                cls_label[cls_label == 32001] = 1
                # [데이터 전처리 부분 끝]

                # Gate(Mistral) 실행 및 시간 측정
                if input_embed.is_cuda: torch.cuda.synchronize()
                t_gate_start = time.perf_counter()
                
                cls_output = self.cls_net(input_embed, cls_labels=cls_label, cls_attention_mask=cls_attention_mask)
                
                if input_embed.is_cuda: torch.cuda.synchronize()
                self.last_gate_ms = (time.perf_counter() - t_gate_start) * 1000

                if cls_training:
                    return cls_output
                else:
                    gate_logits = cls_output.logits[:, -1, :]  
                    return x, gate_logits[-1]

            # --- 경로 B: 기본 프레임 피처 입력 (단순 EOS/Caption 토큰) ---
            else:
                pad_token_id = 0
                input_embeds = []
                cls_labels = []
                start_feature_idx = [0] + frames_features_shape[:-1]
                for idx, end_frame_idx in enumerate(frames_features_shape):
                    cur_frame_feature = x[0][start_feature_idx[idx] : end_frame_idx]
                    if cur_frame_feature.shape[0] == 0: continue
                    eos_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([0]).to(x.device))
                    caption_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([1]).to(x.device))
                    ignore_tensor = torch.tensor([IGNORE_INDEX]).to(x.device)

                    if cur_frame_feature.shape[0] > 1:
                        input_embed = torch.cat([torch.cat([frame.unsqueeze(0), eos_target]) for frame in cur_frame_feature[:-1]])
                        eos_label = torch.cat([torch.cat([ignore_tensor, torch.tensor([0]).to(x.device)]) for _ in cur_frame_feature[:-1]])
                        input_embeds.append(torch.cat([input_embed, cur_frame_feature[-1].unsqueeze(0), caption_target]))
                        cls_labels.append(torch.cat([eos_label, ignore_tensor, torch.tensor([1]).to(x.device)]))
                    else:
                        input_embed = torch.cat([cur_frame_feature, caption_target])
                        caption_label = torch.cat([ignore_tensor, torch.tensor([1]).to(x.device)])
                        input_embeds.append(input_embed)
                        cls_labels.append(caption_label)
                
                input_embeds = torch.cat(input_embeds)
                cls_labels = torch.cat(cls_labels)
                input_embed = einops.rearrange(input_embeds, "(b t) c -> b t c", t=2)
                cls_label = einops.rearrange(cls_labels, "(b t) -> b t", t=2)
                if cls_label.shape[0] > 4000:
                    cls_label = cls_label[:4000]
                    input_embed = input_embed[:4000]
                cls_attention_mask = (input_embed.abs().sum(dim=-1) != 0).to(dtype=torch.long)

                # Gate(Mistral) 실행 및 시간 측정 (수정됨)
                if input_embed.is_cuda: torch.cuda.synchronize()
                t_gate_start = time.perf_counter()

                cls_output = self.cls_net(input_embed, cls_labels=cls_label, cls_attention_mask=cls_attention_mask)
                
                if input_embed.is_cuda: torch.cuda.synchronize()
                self.last_gate_ms = (time.perf_counter() - t_gate_start) * 1000

                if cls_training:
                    return cls_output
                else:
                    gate_logits = cls_output.logits[:, -1, :]   
                    return x, gate_logits[-1]

        # --- 경로 C: 데모 모드 (단일 프레임/마지막 특징) ---
        if cls_demo:
            pad_token_id = 0
            input_embeds = [x[0][-1].unsqueeze(0)]
            input_embed = torch.nn.utils.rnn.pad_sequence(input_embeds, batch_first=True, padding_value=pad_token_id)
            cls_attention_mask = (input_embed.abs().sum(dim=-1) != 0).to(dtype=torch.long)

            # Gate(Mistral) 실행 및 시간 측정 (수정됨)
            if input_embed.is_cuda: torch.cuda.synchronize()
            t_gate_start = time.perf_counter()

            cls_output = self.cls_net(input_embed, cls_labels=None, cls_attention_mask=cls_attention_mask)
            
            if input_embed.is_cuda: torch.cuda.synchronize()
            self.last_gate_ms = (time.perf_counter() - t_gate_start) * 1000
            
            return x, cls_output.logits[0][-1]

        return x

def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class STCConnector(nn.Module):
    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.downsample = downsample
        if depth != 0:
            self.s1 = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1 = nn.Identity()
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=1,
                bias=True,
            ),
            nn.SiLU(),
        )
        if depth != 0:
            self.s2 = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2 = nn.Identity()
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        self.cls_net = ClsNet( d_model=hidden_size, depth=4, n_class=2)
        self.time_list = []
        self.videoid = 0 
        self.conv_fp32 = os.environ.get("STREAMMIND_STC_FP32", "0") == "1"
        if self.conv_fp32:
            self.s1 = self.s1.float()
            self.s2 = self.s2.float()
            self.sampler = self.sampler.float()
    
    def step(self, x, state=None, frames_features_shape=None):
        """
        STCConnector용 O(1) 증분 처리 메서드 (타이머 포함)
        """
        # [타이머 시작] STC 연산 (Perception Token 생성)
        if x.is_cuda: torch.cuda.synchronize()
        t_stc_start = time.perf_counter()

        b, t, n, h = x.shape
        hw = int(n**0.5)

        # 1. Spatial Stage 1
        x_s = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        
        # 2D Conv
        x_s = einops.rearrange(x_s, "b d t h w -> (b t) d h w")
        x_s = self.s1(x_s)
        x_s = einops.rearrange(x_s, "(b t) d h w -> b d t h w", t=t)

        # 2. Sampler (Conv3d)
        x_s = self.sampler(x_s)

        # 3. Spatial Stage 2
        x_s = einops.rearrange(x_s, "b d t h w -> (b t) d h w")
        x_s = self.s2(x_s)
        x_s = einops.rearrange(x_s, "(b t) d h w -> b (t h w) d", t=t) 

        # 4. Readout (MLP)
        readout_dtype = next(self.readout.parameters()).dtype
        x_s = x_s.to(dtype=readout_dtype)
        x_out = self.readout(x_s)

        # [타이머 종료] STC 시간 기록
        if x.is_cuda: torch.cuda.synchronize()
        self.last_mamba_ms = (time.perf_counter() - t_stc_start) * 1000

        # [타이머 시작] Gate 연산 (Mistral)
        t_gate_start = time.perf_counter()

        # 5. Gate (ClsNet)
        eos_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([0], device=x.device)).view(1, 1, -1)
        input_embed = torch.cat([x_out, eos_target.expand(b, -1, -1)], dim=1)
        
        cls_output = self.cls_net(input_embed, cls_labels=None, cls_attention_mask=None)
        cls_last = cls_output.logits[:, -1, :]

        # [타이머 종료] Gate 시간 기록
        if x.is_cuda: torch.cuda.synchronize()
        self.last_gate_ms = (time.perf_counter() - t_gate_start) * 1000

        return x_out, cls_last, state

    def forward(self, x, cls_inference=False, cls_training=False, cls_demo=False, frames_features_shape=[]):
        # 0. 시간 저장 변수 초기화
        self.last_mamba_ms = 0.0  # STC 연산 시간 (그래프의 Mamba SSM 항목에 대응)
        self.last_gate_ms = 0.0   # Gate(Mistral) 연산 시간
        
        t = x.size(1)
        in_dtype = x.dtype

        # -------------------------------------------------------
        # 1. STC (Spatial-Temporal Connector) 연산 시작
        # -------------------------------------------------------
        if x.is_cuda: torch.cuda.synchronize()
        t_stc_start = time.perf_counter()

        if getattr(self, "conv_fp32", False):
            x = x.float()

        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w").contiguous()

        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)

        x = self.sampler(x)
        new_t = x.size(2)

        x = einops.rearrange(x, "b d t h w -> (b t) d h w").contiguous()
        x = self.s2(x)

        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        readout_dtype = next(self.readout.parameters()).dtype
        x = x.to(dtype=readout_dtype)
        x = self.readout(x)

        if getattr(self, "conv_fp32", False):
            x = x.to(dtype=in_dtype)

        if x.is_cuda: torch.cuda.synchronize()
        self.last_mamba_ms = (time.perf_counter() - t_stc_start) * 1000
        # -------------------------------------------------------

        self.videoid += 1
        dump_enable = os.environ.get("STREAMMIND_DUMP_ENABLE", "0") == "1"
        dump_root = os.environ.get("STREAMMIND_DUMP_DIR", "/tmp/streammind_dump")

        if dump_enable:
            out_dir = os.path.join(dump_root, f"immediate_memory_stc_{self.videoid}")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(x, os.path.join(out_dir, "cur_frame_feature.pt"))
   
        if cls_training or cls_inference:
            # Gate 입력을 위한 전처리 (Tokenization/Embedding 과정)
            pad_token_id = 0
            input_embeds = []
            cls_labels = []
            start_feature_idx = [0] + frames_features_shape[:-1]

            for idx, end_frame_idx in enumerate(frames_features_shape):
                start = start_feature_idx[idx]
                if end_frame_idx <= start:
                    continue

                cur_frame_feature = x[0][start:end_frame_idx]

                # (Dump 로직 생략 가능 - 기존 유지)
                if os.environ.get("STREAMMIND_DUMP", "0") == "1":
                    dump_dir = os.environ.get("STREAMMIND_DUMP_DIR", os.path.join(os.path.expanduser("~"), "streammind_dumps", f"immediate_memory_stc_{self.videoid}"))
                    os.makedirs(dump_dir, exist_ok=True)
                    torch.save(cur_frame_feature, os.path.join(dump_dir, f"cur_frame_feature_{idx}.pt"))

                eos_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([0]).to(x.device))
                caption_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([1]).to(x.device))
                ignore_tensor = torch.tensor([IGNORE_INDEX]).to(x.device)

                if cur_frame_feature.shape[0] > 1:
                    input_embed = torch.cat([torch.cat([frame.unsqueeze(0), eos_target]) for frame in cur_frame_feature[:-1]])
                    eos_label = torch.cat([torch.cat([ignore_tensor, torch.tensor([0]).to(x.device)]) for _ in cur_frame_feature[:-1]])
                    input_embeds.append(torch.cat([input_embed, cur_frame_feature[-1].unsqueeze(0), caption_target]))
                    cls_labels.append(torch.cat([eos_label, ignore_tensor, torch.tensor([1]).to(x.device)]))
                else:
                    input_embed = torch.cat([cur_frame_feature, caption_target])
                    caption_label = torch.cat([ignore_tensor, torch.tensor([1]).to(x.device)])
                    input_embeds.append(input_embed)
                    cls_labels.append(caption_label)

            if len(input_embeds) == 0:
                return x, torch.tensor([1.0, -1.0], device=x.device)

            input_embeds = torch.cat(input_embeds)  
            cls_labels = torch.cat(cls_labels)     

            if input_embeds.shape[0] % 2 == 1:
                pad_embed = torch.zeros((1, input_embeds.shape[1]), device=input_embeds.device, dtype=input_embeds.dtype)
                input_embeds = torch.cat([input_embeds, pad_embed], dim=0)
                cls_labels = torch.cat([cls_labels, torch.tensor([IGNORE_INDEX], device=cls_labels.device, dtype=cls_labels.dtype)], dim=0)

            input_embed = einops.rearrange(input_embeds, "(b t) c -> b t c", t=2)
            cls_label  = einops.rearrange(cls_labels, "(b t) -> b t", t=2)

            if cls_label.shape[0] > 4000:
                cls_label = cls_label[:4000]
                input_embed = input_embed[:4000]
            cls_attention_mask = (input_embed.abs().sum(dim=-1) != 0).to(dtype=torch.long)

            # -------------------------------------------------------
            # 2. Gate (Mistral Transformer) 연산 시작
            # -------------------------------------------------------
            if input_embed.is_cuda: torch.cuda.synchronize()
            t_gate_start = time.perf_counter()

            if cls_training:
                cls_output = self.cls_net(input_embed, cls_labels=cls_label, cls_attention_mask=cls_attention_mask)
                
                if input_embed.is_cuda: torch.cuda.synchronize()
                self.last_gate_ms = (time.perf_counter() - t_gate_start) * 1000
                return cls_output
            else:
                cls_output = self.cls_net(input_embed, cls_labels=cls_label, cls_attention_mask=cls_attention_mask)
                
                if input_embed.is_cuda: torch.cuda.synchronize()
                self.last_gate_ms = (time.perf_counter() - t_gate_start) * 1000
                
                gate_logits = cls_output.logits[:, -1, :] 
                gate_logits = gate_logits[-1]          
                return x, gate_logits

        if cls_demo:
            # Demo 모드 Gate 측정
            pad_token_id = 0
            input_embeds = []
            start_feature_idx = [0] + frames_features_shape[:-1]
            if len(frames_features_shape) == 0:
                input_embeds.append(x[0])
            else:
                cur_frame_feature = x[0][frames_features_shape[-1]:]
                eos_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([0]).to(x.device))
                if cur_frame_feature.shape[0] > 1:
                    input_embed = torch.cat([torch.cat([frame.unsqueeze(0),eos_target]) for frame in cur_frame_feature[:-1]])
                    input_embeds.append(torch.cat([input_embed, cur_frame_feature[-1].unsqueeze(0)]))
                else:
                    input_embeds.append(cur_frame_feature)

            input_embed = torch.nn.utils.rnn.pad_sequence(input_embeds, batch_first=True, padding_value=pad_token_id)
            cls_attention_mask = torch.ones((input_embed.shape[0], input_embed.shape[1]), device=input_embed.device, dtype=torch.long)
            
            if input_embed.is_cuda: torch.cuda.synchronize()
            t_demo_start = time.perf_counter()
            
            cls_output = self.cls_net(input_embed, cls_labels=None, cls_attention_mask=cls_attention_mask)
            
            if input_embed.is_cuda: torch.cuda.synchronize()
            self.last_gate_ms = (time.perf_counter() - t_demo_start) * 1000
            
            return x, cls_output.logits[0][-1]
            
        return x

class STPConnector(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(
            config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth
        )
        self.sampler = nn.Sequential(nn.AvgPool3d(downsample), nn.SiLU())

class STCConnectorV35(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(
            config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth
        )
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=0,
                bias=True,
            ),
            nn.SiLU(),
        )

class SpatialConv(STCConnector):

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(
            config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth
        )

class SpatialPool(STPConnector):

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(
            config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth
        )
