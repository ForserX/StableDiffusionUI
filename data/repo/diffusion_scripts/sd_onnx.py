import os
import sys
import time
import torch
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionInpaintPipeline, OnnxRuntimeModel
from diffusers import EulerAncestralDiscreteScheduler, PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, DDPMScheduler, KDPM2DiscreteScheduler, HeunDiscreteScheduler

import argparse
from PIL import PngImagePlugin, Image

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Path to model checkpoint file",
    dest='mdlpath',
)

parser.add_argument(
    "--width",
    type=int,
    help="Path to model checkpoint file",
    dest='width',
)

parser.add_argument(
    "--guidance_scale",
    type=float,
    help="Path to model checkpoint file",
    dest='guidance_scale',
)

parser.add_argument(
    "--height",
    type=int,
    help="Path to model checkpoint file",
    dest='height',
)

parser.add_argument(
    "--totalcount",
    type=int,
    help="Path to model checkpoint file",
    dest='totalcount',
)

parser.add_argument(
    "--steps",
    type=int,
    help="Path to model checkpoint file",
    dest='steps',
)

parser.add_argument(
    "--seed",
    type=int,
    help="Path to model checkpoint file",
    dest='seed',
)

parser.add_argument(
    "--imgscale",
    type=float,
    default=0.44,
    help="Path to model checkpoint file",
    dest='imgscale',
)

parser.add_argument(
    "--prompt_neg",
    type=str,
    help="Path to model checkpoint file",
    dest='prompt_neg',
)

parser.add_argument(
    "--prompt",
    type=str,
    help="Path to model checkpoint file",
    dest='prompt',
)

parser.add_argument(
    "--outpath",
    type=str,
    help="Output path",
    dest='outpath',
)
parser.add_argument(
    "--mode",
    choices=['txt2img', 'img2img', 'inpaint'],
    default="txt2img",
    help="Specify generation mode",
    dest='mode',
)

parser.add_argument(
    "--img",
    type=str,
    default="",
    help="Specify generation mode",
    dest='img',
)

parser.add_argument(
    "--device",
    choices=['dml', 'cuda', 'cpu'],
    default="dml",
    help="Specify generation device",
    dest='device',
)

parser.add_argument(
    "--scmode",
    default="eulera",
    help="Specify generation scmode",
    dest='scmode',
)

parser.add_argument(
    "--vae",
    help="Specify generation vae path",
    dest='vae',
)

parser.add_argument(
    "--eta",
    help="Eta",
    dest='eta',
    default=1.0,
)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()
eta = opt.eta

if opt.device == "dml":
    prov = "DmlExecutionProvider"
if opt.device == "cuda":
    prov = "CUDAExecutionProvider"
if opt.device == "cpu":
    prov = "CPUExecutionProvider"
    
if opt.vae == "default":
    cpuvae = OnnxRuntimeModel.from_pretrained(opt.mdlpath + "/vae_decoder", provider=prov)
else:
    cpuvae = OnnxRuntimeModel.from_pretrained(opt.vae + "/vae_decoder", provider=prov)

# TextEnc moved to CPUs
cputextenc = OnnxRuntimeModel.from_pretrained(opt.mdlpath+"/text_encoder", provider="CPUExecutionProvider")

if opt.mode == "txt2img":
    pipe = OnnxStableDiffusionPipeline.from_pretrained(opt.mdlpath, provider=prov, custom_pipeline="lpw_stable_diffusion_onnx", revision="onnx", text_encoder=cputextenc, vae_decoder=cpuvae, safety_checker=None)
if opt.mode == "img2img":
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(opt.mdlpath, provider=prov, custom_pipeline="lpw_stable_diffusion_onnx", revision="onnx", text_encoder=cputextenc, vae_decoder=cpuvae, safety_checker=None)
if opt.mode == "inpaint":
    pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(opt.mdlpath, provider=prov, custom_pipeline="lpw_stable_diffusion_onnx", revision="onnx", text_encoder=cputextenc, vae_decoder=cpuvae, safety_checker=None)

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
if opt.scmode == "DDPM":
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
if opt.scmode == "DPMDiscrete":
    pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
if opt.scmode == "HeunDiscrete":
    pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)

def generate(prompt, prompt_neg, steps, width, height, seed, scale, init_img_path = None, init_strength = 0.75, mask_img_path = None):
    start_time = time.time()
    
    seed = int(seed)
    print(f"Set seed to {seed}", flush=True)
    
    info = PngImagePlugin.PngInfo()
    neg_prompt_meta_text = "" if prompt_neg == "" else f' [{prompt_neg}]'
        
    rng = torch.Generator(device="cpu").manual_seed(seed)
    
    if opt.mode == "txt2img":
        print("txt2img", flush=True)
        image=pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, generator=rng).images[0]
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

for i in range(opt.totalcount):
    generate(opt.prompt, opt.prompt_neg, opt.steps, opt.width, opt.height, opt.seed, opt.guidance_scale, opt.img , opt.imgscale, "")
    opt.seed = opt.seed + 1
    
print("SD: Generating done!")
