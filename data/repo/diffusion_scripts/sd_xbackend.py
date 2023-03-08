import torch, time, os
from PIL import PngImagePlugin, Image

from pipeline_onnx_stable_diffusion_instruct_pix2pix import OnnxStableDiffusionInstructPix2PixPipeline

from safetensors.torch import load_file

from diffusers import ( 
    OnnxStableDiffusionPipeline, 
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline, 
    OnnxRuntimeModel,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline, 
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline, 
    StableDiffusionInstructPix2PixPipeline,
    ControlNetModel
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
        print ("IsONNX: ", IsONNX)  
        if Mode == "pix2pix":
            if NSFW:
                pipe = OnnxStableDiffusionInstructPix2PixPipeline.from_pretrained("ForserX/instruct-pix2pix-onnx", custom_pipeline="lpw_stable_diffusion_onnx", provider=prov)
            else:
                pipe = OnnxStableDiffusionInstructPix2PixPipeline.from_pretrained("ForserX/instruct-pix2pix-onnx", custom_pipeline="lpw_stable_diffusion_onnx", provider=prov, safety_checker=None)
        else:
            if NSFW:
                safety_model = Model + "/safety_checker/"
                nsfw_pipe = OnnxRuntimeModel.from_pretrained(safety_model, provider=prov)
            print (Mode)    
            if Mode == "txt2img":
                pipe = OnnxStableDiffusionPipeline.from_pretrained(Model, custom_pipeline="lpw_stable_diffusion_onnx", provider=prov, safety_checker=nsfw_pipe)
            if Mode == "img2img":
                pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(Model, custom_pipeline="lpw_stable_diffusion_onnx", provider=prov, safety_checker=nsfw_pipe)
            if Mode == "inpaint":
                pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(Model, custom_pipeline="lpw_stable_diffusion_onnx", provider=prov, safety_checker=nsfw_pipe)
    else:
        print ("IsONNX: ", IsONNX) 
        if fp16:
            fptype = torch.float16
        else:
            fptype = torch.float32

        if Mode == "pix2pix":
            if NSFW:
                pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=fptype)
            else:
                pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=fptype, safety_checker=None)
        else:
            if NSFW:
                safety_model = Model + "/safety_checker/"
                nsfw_pipe = StableDiffusionSafetyChecker.from_pretrained( safety_model, torch_dtype=fptype)
            print (Mode)      
            if Mode == "txt2img":
                pipe = StableDiffusionPipeline.from_pretrained(Model, custom_pipeline="lpw_stable_diffusion", torch_dtype=fptype, safety_checker=nsfw_pipe)
            if Mode == "img2img":
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(Model, custom_pipeline="lpw_stable_diffusion", torch_dtype=fptype, safety_checker=nsfw_pipe)
            if Mode == "inpaint":
                pipe = StableDiffusionInpaintPipeline.from_pretrained(Model, custom_pipeline="lpw_stable_diffusion", torch_dtype=fptype, safety_checker=nsfw_pipe)


    return pipe

def GetPipeCN(Model: str, CNModel: str, NSFW: bool, fp16: bool):
    pipe = None
    nsfw_pipe = None
    
    if fp16:
        fptype = torch.float16
    else:
        fptype = torch.float32

    controlnet = ControlNetModel.from_pretrained(
        CNModel, torch_dtype=fptype
    )

    if NSFW:
        safety_model = Model + "/safety_checker/"
        nsfw_pipe = StableDiffusionSafetyChecker.from_pretrained(safety_model, torch_dtype=fptype)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        Model,
        controlnet=controlnet,
        torch_dtype=fptype, 
        safety_checker=nsfw_pipe
    )

    return pipe

def ApplyLoRA(pipe, LoraPath : str, device, fp16: bool):
    model_path = LoraPath
    state_dict = load_file(model_path, device)
    
    if fp16:
        fptype = torch.float16
    else:
        fptype = torch.float32


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

def MakeImage(pipe, mode : str, eta, prompt, prompt_neg, steps, width, height, seed, scale, image_guidance_scale, device = "cuda", init_img_path = None, img_strength = 0.75, mask_img_path = None, outpath = ""):
    start_time = time.time()
    
    seed = int(seed)
    print(f"Set seed to {seed}", flush=True)
    
    info = PngImagePlugin.PngInfo()
    neg_prompt_meta_text = "" if prompt_neg == "" else f' [{prompt_neg}]'
        
    rng = torch.Generator(device).manual_seed(seed)
    print(mode, flush=True)
    
    if mode == "txt2img":
        image=pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale}')
    
    if mode == "img2img":
        # Opt image
        img=Image.open(init_img_path).convert("RGB").resize((width, height))
        image=pipe(prompt=prompt, image=img, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, strength=img_strength, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale} -I {init_img_path} -f {img_strength}')

    if mode == "pix2pix":
        # Opt image       

        img=Image.open(init_img_path).convert("RGB").resize((width, height))
        image=pipe(prompt=prompt, image=img, num_inference_steps=steps, guidance_scale=scale, image_guidance_scale=image_guidance_scale, negative_prompt=prompt_neg, eta=eta, generator=rng).images[0]
        info.add_text('Dream',  f'"{prompt}{neg_prompt_meta_text}" -s {steps} -S {seed} -W {width} -H {height} -C {scale} -I {init_img_path} -f {img_strength}')

    if mode == "inpaint":
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
        "--mode", choices=['txt2img', 'img2img', 'inpaint', 'pix2pix', 'IfP', 'PfI'], default="txt2img", help="Specify generation mode", dest='mode',
    )
    parser.add_argument(
        "--img", type=str, default=None, help="Specify generation mode", dest='img',
    )
    parser.add_argument(
        "--imgmask", type=str, default=None, help="Specify generation image mask", dest='imgmask',
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
    parser.add_argument(
        "--cn_model", type=str, help="Path to model checkpoint file", dest='cn_model',
    )
    parser.add_argument(
        "--outfile", type=str, default="", help="Specify generation mode", dest='outfile',
    )
    parser.add_argument(
        "--pose", type=str, default="", help="input pose image", dest='pose',
    )
    parser.add_argument(
        "--strenght", type=float, default=0.45, help="strenght", dest='strenght',
    )
    parser.add_argument(
        "--img_strength", type=float, default=0.75, help="img_strength", dest='img_strength',
    )
    parser.add_argument(
        "--image_guidance_scale", type=float, default=1.5, help="image_guidance_scale", dest='image_guidance_scale',
    )