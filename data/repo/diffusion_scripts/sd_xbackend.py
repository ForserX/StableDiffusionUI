import torch, time, os
from PIL import PngImagePlugin, Image

from diffusers import ( 
    OnnxStableDiffusionPipeline, 
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline, 
    OnnxRuntimeModel,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline, 
    StableDiffusionInpaintPipeline
)

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import (
    EulerAncestralDiscreteScheduler, 
    PNDMScheduler, 
    LMSDiscreteScheduler, 
    DDIMScheduler, 
    DPMSolverMultistepScheduler, 
    EulerDiscreteScheduler, 
    DDPMScheduler, 
    KDPM2DiscreteScheduler, 
    HeunDiscreteScheduler
)

prov = "DmlExecutionProvider"

def GetPipe(Model: str, Mode: str, IsONNX: bool, NSFW: bool, fp16: bool):
    pipe = None
    nsfw_pipe = None
    
    if IsONNX:
        if NSFW:
            safety_model = Model + "/safety_checker/"
            nsfw_pipe = OnnxRuntimeModel.from_pretrained(safety_model, provider=prov)
            
        if Mode == "txt2img":
            pipe = OnnxStableDiffusionPipeline.from_pretrained(Model, custom_pipeline="lpw_stable_diffusion_onnx", provider=prov, safety_checker=nsfw_pipe)
        if Mode == "img2img":
            pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(Model, custom_pipeline="lpw_stable_diffusion_onnx", provider=prov, safety_checker=nsfw_pipe)
        if Mode == "img2img":
            pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(Model, custom_pipeline="lpw_stable_diffusion_onnx", provider=prov, safety_checker=nsfw_pipe)
    else:
        if fp16:
            fptype = torch.float16
        else:
            fptype = torch.float32

        if NSFW:
            safety_model = Model + "/safety_checker/"
            nsfw_pipe = StableDiffusionSafetyChecker.from_pretrained( safety_model, torch_dtype=fptype)
            
        if Mode == "txt2img":
            pipe = StableDiffusionPipeline.from_pretrained(Model, custom_pipeline="lpw_stable_diffusion", torch_dtype=fptype, safety_checker=nsfw_pipe)
        if Mode == "img2img":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(Model, custom_pipeline="lpw_stable_diffusion", torch_dtype=fptype, safety_checker=nsfw_pipe)
        if Mode == "img2img":
            pipe = StableDiffusionInpaintPipeline.from_pretrained(Model, custom_pipeline="lpw_stable_diffusion", torch_dtype=fptype, safety_checker=nsfw_pipe)

    return pipe

def GetSampler(Pipe, SamplerName: str, ETA):
    eta = 0
    if SamplerName == "EulerAncestralDiscrete":
        Pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(Pipe.scheduler.config)
        eta = ETA
    if SamplerName == "EulerDiscrete":
        Pipe.scheduler = EulerDiscreteScheduler.from_config(Pipe.scheduler.config)
    if SamplerName == "PNDM":
        Pipe.scheduler = PNDMScheduler.from_config(Pipe.scheduler.config)
    if SamplerName == "DDIM":
        Pipe.scheduler = DDIMScheduler.from_config(Pipe.scheduler.config)
    if SamplerName == "DPMSolverMultistep":
        Pipe.scheduler = DPMSolverMultistepScheduler.from_config(Pipe.scheduler.config)
    if SamplerName == "LMSDiscrete":
        Pipe.scheduler = LMSDiscreteScheduler.from_config(Pipe.scheduler.config)
    if SamplerName == "DDPM":
        Pipe.scheduler = DDPMScheduler.from_config(Pipe.scheduler.config)
    if SamplerName == "DPMDiscrete":
        Pipe.scheduler = KDPM2DiscreteScheduler.from_config(Pipe.scheduler.config)
    if SamplerName == "HeunDiscrete":
        Pipe.scheduler = HeunDiscreteScheduler.from_config(Pipe.scheduler.config)

    return eta

def MakeImage(pipe, mode : str, eta, prompt, prompt_neg, steps, width, height, seed, scale, init_img_path = None, init_strength = 0.75, mask_img_path = None, outpath = ""):
    start_time = time.time()
    
    seed = int(seed)
    print(f"Set seed to {seed}", flush=True)
    
    info = PngImagePlugin.PngInfo()
    neg_prompt_meta_text = "" if prompt_neg == "" else f' [{prompt_neg}]'
        
    rng = torch.Generator(device="cpu").manual_seed(seed)
    
    if mode == "txt2img":
        print("txt2img", flush=True)
        image=pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale}')
        
    if mode == "img2img":
        print("img2img", flush=True)
        # Opt image
        img=Image.open(init_img_path).convert("RGB").resize((width, height))
        
        image=pipe(prompt=prompt, image=img, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, strength=init_strength, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale} -I {init_img_path} -f {init_strength}')
    if mode == "inpaint":
        print("inpaint", flush=True)

        img=Image.open(init_img_path).convert("RGB").resize((width, height))
        mask=Image.open(mask_img_path).convert("RGB").resize((width, height))

        image=pipe(prompt=prompt, image=img, mask_image = mask, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale} -I {init_img_path} -f 0.0 -M {mask_img_path}')

    image.save(os.path.join(outpath, f"{time.time_ns()}.png"), 'PNG', pnginfo=info)
    
    print(f'Image generated in {(time.time() - start_time):.2f}s')
    image = None

def ApplyArg(parser):
    parser.add_argument(
        "--model", type=str, help="Path to model checkpoint file", dest='mdlpath',
    )
    parser.add_argument(
        "--width", type=int, help="Path to model checkpoint file", dest='width',
    )
    parser.add_argument(
        "--guidance_scale", type=float, help="Path to model checkpoint file", dest='guidance_scale',
    )
    parser.add_argument(
        "--height", type=int, help="Path to model checkpoint file", dest='height',
    )
    parser.add_argument(
        "--totalcount",
        type=int, help="Path to model checkpoint file", dest='totalcount',
    )
    parser.add_argument(
        "--steps",
        type=int, help="Path to model checkpoint file", dest='steps',
    )
    parser.add_argument(
        "--seed", type=int, help="Path to model checkpoint file", dest='seed',
    )
    parser.add_argument(
        "--imgscale",
        type=float, default=0.44, help="Path to model checkpoint file", dest='imgscale',
    )
    parser.add_argument(
        "--prompt_neg", type=str, help="Path to model checkpoint file", dest='prompt_neg',
    )
    parser.add_argument(
        "--prompt", type=str, help="Path to model checkpoint file", dest='prompt',
    )
    parser.add_argument(
        "--outpath",
        type=str, help="Output path", dest='outpath',
    )
    parser.add_argument(
        "--precision", type=str, help="precision type (fp16/fp32)", dest='precision',
    )
    parser.add_argument(
        "--mode", choices=['txt2img', 'img2img', 'inpaint'], default="txt2img", help="Specify generation mode", dest='mode',
    )
    parser.add_argument(
        "--img", type=str, default="", help="Specify generation mode", dest='img',
    )
    parser.add_argument(
        "--imgmask", type=str, default="", help="Specify generation image mask", dest='imgmask',
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Specify generation mode device", dest='device',
    )
    parser.add_argument(
        "--scmode", default="eulera", help="Specify generation scmode", dest='scmode',
    )
    parser.add_argument(
        "--vae", help="Specify generation vae path", dest='vae',
    )
    parser.add_argument(
        "--eta", help="Eta", dest='eta', default=1.0,
    )
    parser.add_argument(
        "--nsfw", help="nsfw checker", dest='nsfw', default=False,
    )
    parser.add_argument(
        "--lora", help="lora checker", dest='lora', default=False,
    )
    parser.add_argument(
        "--lora_path", type=str, help="Path to model LoRA file", dest='lora_path',
    )
    parser.add_argument(
        "--inversion", help="inversion path", dest='inversion', default=None,
    )