import os
import sys
import time
import torch
import numpy as np
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionInpaintPipeline
from diffusers import EulerAncestralDiscreteScheduler, PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, DDPMScheduler

import argparse
from PIL import PngImagePlugin, Image

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="Path to model checkpoint file",
    dest='mdlpath',
)

parser.add_argument(
    "-width",
    "--width",
    type=int,
    help="Path to model checkpoint file",
    dest='width',
)

parser.add_argument(
    "-guidance_scale",
    "--guidance_scale",
    type=float,
    help="Path to model checkpoint file",
    dest='guidance_scale',
)

parser.add_argument(
    "-height",
    "--height",
    type=int,
    help="Path to model checkpoint file",
    dest='height',
)

parser.add_argument(
    "-totalcount",
    "--totalcount",
    type=int,
    help="Path to model checkpoint file",
    dest='totalcount',
)

parser.add_argument(
    "-steps",
    "--steps",
    type=int,
    help="Path to model checkpoint file",
    dest='steps',
)

parser.add_argument(
    "-seed",
    "--seed",
    type=int,
    help="Path to model checkpoint file",
    dest='seed',
)

parser.add_argument(
    "-prompt_neg",
    "--prompt_neg",
    type=str,
    help="Path to model checkpoint file",
    dest='prompt_neg',
)

parser.add_argument(
    "-prompt",
    "--prompt",
    type=str,
    help="Path to model checkpoint file",
    dest='prompt',
)

parser.add_argument(
    "-o",
    "--outpath",
    type=str,
    help="Output path",
    dest='outpath',
)
parser.add_argument(
    "-mode",
    "--mode",
    choices=['txt2img', 'img2img', 'inpaint'],
    default="txt2img",
    help="Specify generation mode",
    dest='mode',
)

parser.add_argument(
    "-device",
    "--device",
    choices=['dml', 'cuda', 'cpu'],
    default="dml",
    help="Specify generation device",
    dest='device',
)

parser.add_argument(
    "-scmode",
    "--scmode",
    choices=['EulerAncestralDiscrete', 'EulerDiscrete', 'PNDM', 'DPMSolverMultistep', 'LMSDiscrete', 'SharkEulerDiscrete', 'DDIM'],
    default="eulera",
    help="Specify generation scmode",
    dest='scmode',
)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()
eta = 0.0

if opt.device == "dml":
    prov = "DmlExecutionProvider"
if opt.device == "cuda":
    prov = "CUDAExecutionProvider"
if opt.device == "cpu":
    prov = "CPUExecutionProvider"

if opt.mode == "txt2img":
    pipe = OnnxStableDiffusionPipeline.from_pretrained(opt.mdlpath, provider=prov, safety_checker=None)
if opt.mode == "img2img":
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(opt.mdlpath, provider=prov, revision="onnx", safety_checker=None)
if opt.mode == "inpaint":
    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(opt.mdlpath, provider=prov, revision="onnx", safety_checker=None)

if opt.scmode == "EulerAncestralDiscrete":
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
if opt.scmode == "EulerDiscrete":
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
if opt.scmode == "PNDM":
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
if opt.scmode == "DDIM":
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
if opt.scmode == "DPMSolverMultistep":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
if opt.scmode == "LMSDiscrete":
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
if opt.scmode == "SharkEulerDiscrete":
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

def generate(prompt, prompt_neg, steps, width, height, seed, scale, init_img_path = None, init_strength = 0.75, mask_img_path = None):
    start_time = time.time()
    
    seed = int(seed)
    rng = np.random.RandomState(seed)
    print(f"Set seed to {seed}", flush=True)
    
    info = PngImagePlugin.PngInfo()
    neg_prompt_meta_text = "" if prompt_neg == "" else f' [{prompt_neg}]'
        
    if opt.mode == "txt2img":
        print("txt2img", flush=True)
        image=pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale}')
        
    if opt.mode == "img2img":
        print("img2img", flush=True)
        img=Image.open(init_img_path)
        
        image=pipe(prompt=prompt, image=img, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, strength=init_strength, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale} -I {init_img_path} -f {init_strength}')
    if opt.mode == "inpaint":
        print("inpaint", flush=True)
        img=Image.open(init_img_path)
        mask=Image.open(mask_img_path)
        
        image=pipe(prompt=prompt, image=img, mask_image = mask, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale} -I {init_img_path} -f 0.0 -M {mask_img_path}')

    image.save(os.path.join(opt.outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo=info)
    
    print(f'Image generated in {(time.time() - start_time):.2f}s')
    image = None

print(f'Model loaded.')

for i in range(opt.totalcount):
#    print(f'Generating {i+1}/{len(data)}: "{argdict["prompt"]}" - {argdict["steps"]} Steps - Scale {argdict["scale"]} - {argdict["w"]}x{argdict["h"]}')
    generate(opt.prompt, opt.prompt_neg, opt.steps, opt.width, opt.height, opt.seed, opt.guidance_scale, "" , 1.0, "")