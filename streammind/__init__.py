import copy
from functools import partial

import torch

from .model import Videollama2LlamaForCausalLM, Videollama2MistralForCausalLM, Videollama2MixtralForCausalLM
from .model.builder import load_pretrained_model
from .conversation import conv_templates, SeparatorStyle
from .mm_utils import process_video, tokenizer_MMODAL_token, get_model_name_from_path, KeywordsStoppingCriteria
from .constants import NUM_FRAMES, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX


def model_init(model_path=None, model_name="VideoLLaMA2-7B"):
    model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path) if model_name is None else model_name
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)

    # pad_token 안정화
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


def infer(model, video, instruct, tokenizer, do_sample=False, version="llama_2", **kwargs):
    """
    변경 핵심:
    - 멀티모달(generate에 images_or_videos를 넣는 경우)에는 attention_mask를 넘기지 않는다.
      (multimodal embedding 삽입 후 길이가 달라져 mask/pos 불일치가 RoPE CUDA assert를 유발)
    """

    # 1) vision preprocess
    tensor = [video.half().cuda()]
    modals = ["video"]

    # 2) text preprocess
    modal_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_index = MMODAL_TOKEN_INDEX["VIDEO"]
    instruct = modal_token + "\n" + instruct

    conv = conv_templates[version].copy()
    conv.append_message(conv.roles[0], instruct)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_MMODAL_token(
        prompt, tokenizer, modal_index, return_tensors="pt"
    ).unsqueeze(0).cuda()

    # stopping criteria는 "초기 input_ids"가 필요하니 유지
    stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # pad_token_id 안정화
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # (선택) text-only일 때만 쓸 mask. 멀티모달에는 넘기지 않을 것.
    attention_masks = input_ids.ne(pad_id).long()

    with torch.inference_mode():
        inputs = dict(
            input_ids=input_ids,
            # ✅ 핵심: 멀티모달에서는 attention_mask를 넘기지 말 것
            attention_mask=None,
            images_or_videos=tensor,
            modal_list=modals,
            do_sample=do_sample,
            temperature=0.2 if do_sample else 0.0,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )
        inputs.update(kwargs)
        output_ids = model.generate(**inputs)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def x_infer(video, question, model, tokenizer, mode="vanilla", do_sample=False, version="llama_2"):
    if mode == "mcqa":
        instruction = (
            f"{question}\nAnswer with the option's letter from the given choices directly and only give the best option."
        )
        return infer(model=model, tokenizer=tokenizer, video=video, instruct=instruction, do_sample=do_sample, version=version)
    elif mode == "openend":
        instruction = f"{question}\nAnswer the question using a single word or a short phrase with multiple words."
        return infer(model=model, tokenizer=tokenizer, video=video, instruct=instruction, do_sample=do_sample, version=version)
    elif mode == "vanilla":
        return infer(model=model, tokenizer=tokenizer, video=video, instruct=question, do_sample=do_sample, version=version)
