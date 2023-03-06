import os
import sys
import time
import torch

from safetensors.torch import load_file

from diffusers import AutoencoderKL

from sd_xbackend import (
    GetPipe,
    GetSampler,
    ApplyArg
)

import argparse
from PIL import PngImagePlugin, Image

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()
ApplyArg(parser)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()

if opt.precision == "fp16":
    fptype = torch.float16
else:
    fptype = torch.float32


pipe = GetPipe(opt.mdlpath, opt.mode, False, opt.nsfw, opt.precision == "fp16")

if not opt.vae == "default":
    pipe.vae = AutoencoderKL.from_pretrained(opt.vae + "/vae", torch_dtype=fptype)
    
pipe.to(opt.device)
eta = GetSampler(pipe, opt.scmode, opt.eta)

if opt.inversion is not None:
    loaded_embeds = torch.load(opt.inversion, opt.device)
    from pathlib import Path

    trained_token = Path(opt.inversion).stem
    if len(loaded_embeds) > 2:
        embeds = loaded_embeds["string_to_param"]["*"][-1, :]
    else:
        # separate token and the embeds
        trained_token = list(loaded_embeds.keys())[0]
        embeds = loaded_embeds[trained_token]

    token = trained_token
    num_added_tokens = pipe.tokenizer.add_tokens(token)
    print(f"Initial token: {trained_token} - {num_added_tokens}")
    i = 1
    while num_added_tokens == 0:
        print(f"The tokenizer already contains the token {token}.")
        token = f"{token[:-1]}-{i}>"
        print(f"Attempting to add the token {token}.")
        num_added_tokens = pipe.tokenizer.add_tokens(token)
        i += 1

    # resize the token embeddings
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

    # get the id for the token and assign the embeds
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    pipe.text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    
    
# LoRA magic
if opt.lora:
    model_path = opt.lora_path
    state_dict = load_file(model_path, opt.device)

    
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'
    
    alpha = 0.75
    
    visited = []
    
    # directly update weight in diffusers model
    for key in state_dict:
        
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
        
        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue
            
        if 'text' in key:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
            curr_layer = pipe.text_encoder
        else:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
            curr_layer = pipe.unet
    
        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += '_'+layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
        
        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))
        
        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(fptype)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(fptype)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(fptype)
            weight_down = state_dict[pair_keys[1]].to(fptype)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
            
         # update visited list
        for item in pair_keys:
            visited.append(item)

def generate(prompt, prompt_neg, steps, width, height, seed, scale, init_img_path = None, init_strength = 0.75, mask_img_path = None):
    start_time = time.time()
    
    seed = int(seed)
    print(f"Set seed to {seed}", flush=True)
    
    rng = torch.Generator(device=opt.device).manual_seed(seed)
    info = PngImagePlugin.PngInfo()
    neg_prompt_meta_text = "" if prompt_neg == "" else f' [{prompt_neg}]'
        
    if opt.mode == "txt2img":
        print("txt2img", flush=True)
        image=pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale}')
        
    if opt.mode == "img2img":
        print("img2img", flush=True)
        # Opt image
        img=Image.open(init_img_path).convert("RGB").resize((width, height))
        
        image=pipe(prompt=prompt, image=img, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, strength=init_strength, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale} -I {init_img_path} -f {init_strength}')
    if opt.mode == "inpaint":
        print("inpaint", flush=True)

        img=Image.open(init_img_path).convert("RGB").resize((width, height))
        mask=Image.open(mask_img_path).convert("RGB").resize((width, height))

        image=pipe(prompt=prompt, image=img, mask_image = mask, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale} -I {init_img_path} -f 0.0 -M {mask_img_path}')

    image.save(os.path.join(opt.outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo=info)
    
    print(f'Image generated in {(time.time() - start_time):.2f}s')
    image = None

print("SD: Model loaded")
print(f"Prompt: {opt.prompt}")
print(f"Neg rompt: {opt.prompt_neg}")

for i in range(opt.totalcount):
    generate(opt.prompt, opt.prompt_neg, opt.steps, opt.width, opt.height, opt.seed, opt.guidance_scale, opt.img , opt.imgscale, opt.imgmask)
    opt.seed = opt.seed + 1
    
print("SD: Generating done!")
