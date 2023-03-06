import os
import sys
import time
import torch

from diffusers import OnnxRuntimeModel
from transformers import CLIPTokenizer

import argparse
from PIL import PngImagePlugin, Image

from sd_xbackend import (
    GetPipe,
    GetSampler,
    ApplyArg
)

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()
ApplyArg(parser)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()
prov = "DmlExecutionProvider"

# TextEnc moved to CPUs

pipe = GetPipe(opt.mdlpath, opt.mode, True, opt.nsfw, False)

if not opt.vae == "default":
    print("Load custom vae")
    pipe.vae_decoder = OnnxRuntimeModel.from_pretrained(opt.vae + "/vae_decoder", provider=prov)
    
if opt.inversion is not None:
    pipe.text_encoder = OnnxRuntimeModel.from_pretrained(opt.mdlpath + "/textual_inversion_merges/" + opt.inversion + "/text_encoder", provider="CPUExecutionProvider")
    pipe.tokenizer = CLIPTokenizer.from_pretrained(opt.mdlpath + "/textual_inversion_merges/" + opt.inversion + "/tokenizer")

eta = GetSampler(pipe, opt.scmode, opt.eta)

def generate(prompt, prompt_neg, steps, width, height, seed, scale, init_img_path = None, init_strength = 0.75, mask_img_path = None):
    start_time = time.time()
    
    seed = int(seed)
    print(f"Set seed to {seed}", flush=True)
    
    info = PngImagePlugin.PngInfo()
    neg_prompt_meta_text = "" if prompt_neg == "" else f' [{prompt_neg}]'
        
    rng = torch.Generator(device="cpu").manual_seed(seed)
    
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
