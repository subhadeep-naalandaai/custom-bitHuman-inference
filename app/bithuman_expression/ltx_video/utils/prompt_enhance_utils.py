# Source Generated with Decompyle++
# File: prompt_enhance_utils.pyc (Python 3.10)

import logging
from typing import Union, List, Optional
import torch
from PIL import Image
logger = logging.getLogger(__name__)
T2V_CINEMATIC_PROMPT = 'You are an expert cinematic director with many award winning movies, When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.\nInclude specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.\nStart directly with the action, and keep descriptions literal and precise.\nThink like a cinematographer describing a shot list.\nDo not change the user input intent, just enhance it.\nKeep within 150 words.\nFor best results, build your prompts using this structure:\nStart with main action in a single sentence\nAdd specific details about movements and gestures\nDescribe character/object appearances precisely\nInclude background and environment details\nSpecify camera angles and movements\nDescribe lighting and colors\nNote any changes or sudden events\nDo not exceed the 150 word limit!\nOutput the enhanced prompt only.\n'
I2V_CINEMATIC_PROMPT = 'You are an expert cinematic director with many award winning movies, When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.\nInclude specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.\nStart directly with the action, and keep descriptions literal and precise.\nThink like a cinematographer describing a shot list.\nKeep within 150 words.\nFor best results, build your prompts using this structure:\nDescribe the image first and then add the user input. Image description should be in first priority! Align to the image caption if it contradicts the user text input.\nStart with main action in a single sentence\nAdd specific details about movements and gestures\nDescribe character/object appearances precisely\nInclude background and environment details\nSpecify camera angles and movements\nDescribe lighting and colors\nNote any changes or sudden events\nAlign to the image caption if it contradicts the user text input.\nDo not exceed the 150 word limit!\nOutput the enhanced prompt only.\n'

def tensor_to_pil(tensor):
    pass
# WARNING: Decompyle incomplete


def generate_cinematic_prompt(image_caption_model, image_caption_processor, prompt_enhancer_model = None, prompt_enhancer_tokenizer = None, prompt = None, conditioning_items = (None, 256), max_new_tokens = ('prompt', Union[(str, List[str])], 'conditioning_items', Optional[List], 'max_new_tokens', int, 'return', List[str])):
    prompts = [
        prompt] if isinstance(prompt, str) else prompt
    if conditioning_items is None:
        prompts = _generate_t2v_prompt(prompt_enhancer_model, prompt_enhancer_tokenizer, prompts, max_new_tokens, T2V_CINEMATIC_PROMPT)
        return prompts
    if None(conditioning_items) > 1 or conditioning_items[0].media_frame_number != 0:
        logger.warning('prompt enhancement does only support unconditional or first frame of conditioning items, returning original prompts')
        return prompts
    first_frame_conditioning_item = None[0]
    first_frames = _get_first_frames_from_conditioning_item(first_frame_conditioning_item)
# WARNING: Decompyle incomplete


def _get_first_frames_from_conditioning_item(conditioning_item = None):
    frames_tensor = conditioning_item.media_item
    return [tensor_to_pil(frames_tensor[i, :, 0, :, :]) for i in range(frames_tensor.shape[0])]


def _generate_t2v_prompt(prompt_enhancer_model, prompt_enhancer_tokenizer = None, prompts = None, max_new_tokens = None, system_prompt = ('prompts', List[str], 'max_new_tokens', int, 'system_prompt', str, 'return', List[str])):
    messages = [[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': f'user_prompt: {p}'}] for p in prompts]
    texts = [prompt_enhancer_tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    model_inputs = prompt_enhancer_tokenizer(texts, 'pt', **('return_tensors',)).to(prompt_enhancer_model.device)
    return _generate_and_decode_prompts(prompt_enhancer_model, prompt_enhancer_tokenizer, model_inputs, max_new_tokens)


def _generate_i2v_prompt(image_caption_model, image_caption_processor, prompt_enhancer_model, prompt_enhancer_tokenizer, prompts = None, first_frames = None, max_new_tokens = None, system_prompt = ('prompts', List[str], 'first_frames', List[Image.Image], 'max_new_tokens', int, 'system_prompt', str, 'return', List[str])):
    image_captions = _generate_image_captions(image_caption_model, image_caption_processor, first_frames)
    messages = [[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': f'user_prompt: {p}\nimage_caption: {c}'}] for p, c in zip(prompts, image_captions)]
    texts = [ prompt_enhancer_tokenizer.apply_chat_template(m, False, True, **('tokenize', 'add_generation_prompt')) for m in messages ]
    model_inputs = prompt_enhancer_tokenizer(texts, 'pt', **('return_tensors',)).to(prompt_enhancer_model.device)
    return _generate_and_decode_prompts(prompt_enhancer_model, prompt_enhancer_tokenizer, model_inputs, max_new_tokens)


def _generate_image_captions(image_caption_model = None, image_caption_processor = None, images = None, system_prompt = ('<DETAILED_CAPTION>',)):
    image_caption_prompts = [
        system_prompt] * len(images)
    inputs = image_caption_processor(image_caption_prompts, images, 'pt', **('return_tensors',)).to(image_caption_model.device)
    with torch.inference_mode():
        generated_ids = image_caption_model.generate(inputs['input_ids'], inputs['pixel_values'], 1024, False, 3, **('input_ids', 'pixel_values', 'max_new_tokens', 'do_sample', 'num_beams'))
        None(None, None, None)
    with None:
        if not None:
            pass
    return image_caption_processor.batch_decode(generated_ids, True, **('skip_special_tokens',))


def _generate_and_decode_prompts(prompt_enhancer_model = None, prompt_enhancer_tokenizer = None, model_inputs = None, max_new_tokens = ('max_new_tokens', int, 'return', List[str])):
    pass
# WARNING: Decompyle incomplete

