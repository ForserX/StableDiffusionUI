import os
import sys
import time
import torch

from safetensors.torch import load_file
from transformers import CLIPTokenizer, CLIPTextModel

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, AutoencoderKL
from diffusers import EulerAncestralDiscreteScheduler, PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, DDPMScheduler, KDPM2DiscreteScheduler, HeunDiscreteScheduler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

import argparse
from PIL import PngImagePlugin, Image

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()

eta = 0.0


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
    "--precision",
    type=str,
    help="precision type (fp16/fp32)",
    dest='precision',
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
    "--imgmask",
    type=str,
    default="",
    help="Specify generation image mask",
    dest='imgmask',
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Specify generation mode device",
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

parser.add_argument(
    "--nsfw",
    help="nsfw checker",
    dest='nsfw',
    default=False,
)

parser.add_argument(
    "--lora",
    help="lora checker",
    dest='lora',
    default=False,
)

parser.add_argument(
    "--lora_path",
    type=str,
    help="Path to model LoRA file",
    dest='lora_path',
)

parser.add_argument(
    "--inversion",
    help="inversion path",
    dest='inversion',
    default=None,
)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()

if opt.precision == "fp16":
    fptype = torch.float16
else:
    fptype = torch.float32

NSFW = None

if opt.nsfw:
    safety_model = opt.mdlpath + "/safety_checker/"
    NSFW = StableDiffusionSafetyChecker.from_pretrained(
        safety_model,
        torch_dtype=fptype
    )

if opt.vae == "default":
    vae = AutoencoderKL.from_pretrained(opt.mdlpath + "/vae", torch_dtype=fptype)
else:
    vae = AutoencoderKL.from_pretrained(opt.vae + "/vae", torch_dtype=fptype)
    
if opt.inversion is not None:
    cputextenc = CLIPTextModel.from_pretrained(opt.mdlpath + "/textual_inversion_merges/" + opt.inversion + "/text_encoder")
    cliptokenizer = CLIPTokenizer.from_pretrained(opt.mdlpath + "/textual_inversion_merges/" + opt.inversion + "/tokenizer")
    
else:
    cputextenc = CLIPTextModel.from_pretrained(opt.mdlpath + "/text_encoder")
    cliptokenizer = CLIPTokenizer.from_pretrained(opt.mdlpath + "/tokenizer")
  
cputextenc.to(opt.device,fptype)

if opt.mode == "txt2img":
    pipe = StableDiffusionPipeline.from_pretrained(opt.mdlpath, custom_pipeline="lpw_stable_diffusion", torch_dtype=fptype, text_encoder=cputextenc, tokenizer=cliptokenizer, vae=vae, safety_checker=NSFW)
if opt.mode == "img2img":
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(opt.mdlpath, custom_pipeline="lpw_stable_diffusion", torch_dtype=fptype, text_encoder=cputextenc, tokenizer=cliptokenizer, vae=vae, safety_checker=NSFW)
if opt.mode == "inpaint":
    pipe = StableDiffusionInpaintPipeline.from_pretrained(opt.mdlpath, custom_pipeline="lpw_stable_diffusion", torch_dtype=fptype, text_encoder=cputextenc, tokenizer=cliptokenizer, vae=vae, safety_checker=NSFW)

pipe.to(opt.device)
cputextenc.to(opt.device, fptype)

if opt.scmode == "EulerAncestralDiscrete":
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, torch_dtype=fptype)
    eta = opt.eta

if opt.scmode == "EulerDiscrete":
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, torch_dtype=fptype)
    
if opt.scmode == "PNDM":
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config, torch_dtype=fptype)
    
if opt.scmode == "DDIM":
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, torch_dtype=fptype)
    
if opt.scmode == "DPMSolverMultistep":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, torch_dtype=fptype)
    
if opt.scmode == "LMSDiscrete":
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config, torch_dtype=fptype)
    
if opt.scmode == "DDPM":
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config, torch_dtype=fptype)
    
if opt.scmode == "DPMDiscrete":
    pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config, torch_dtype=fptype)
    
if opt.scmode == "HeunDiscrete":
    pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config, torch_dtype=fptype)

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

for i in range(opt.totalcount):
    generate(opt.prompt, opt.prompt_neg, opt.steps, opt.width, opt.height, opt.seed, opt.guidance_scale, opt.img , opt.imgscale, opt.imgmask)
    opt.seed = opt.seed + 1
    
print("SD: Generating done!")
