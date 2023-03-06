from diffusers import UniPCMultistepScheduler, AutoencoderKL
import torch, os, time

from controlnet_aux import OpenposeDetector
from PIL import Image
import argparse

from sd_xbackend import (
    GetPipeCN,
    ApplyArg,
    ApplyLoRA
)

parser = argparse.ArgumentParser()
ApplyArg(parser)
opt = parser.parse_args()


if opt.precision == "fp16":
    fptype = torch.float16
else:
    fptype = torch.float32


def generatePoseFromImage():
    print("processing generatePoseFromImage()")
    model = OpenposeDetector.from_pretrained(opt.model)
    in_img = Image.open(opt.img)
    
    img = model(in_img)
    img.save(opt.outfile)
    print(f"CN: Pose - {opt.outfile}")

def generateImageFromPose ():
    print("processing generateImageFromPose()")
    image = Image.open(opt.pose)

    pipe = GetPipeCN(opt.mdlpath, opt.cn_model, opt.nsfw, opt.precision == "fp16")
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if not opt.vae == "default":
        pipe.vae = AutoencoderKL.from_pretrained(opt.vae + "/vae", torch_dtype=fptype)
        
    pipe.to(opt.device)

    #if opt.device == "cpu":
    #    pipe.enable_model_cpu_offload()

    # LoRA magic
    if opt.lora:
        ApplyLoRA(pipe, opt.lora_path, opt.device, opt.precision == "fp16")

    print("CN: Model loaded")
    generator = torch.Generator(device=opt.device).manual_seed(opt.seed)
        
    output = pipe(
        opt.prompt,
        image,
        negative_prompt = opt.prompt_neg,
        generator = generator,
        num_inference_steps=opt.steps
    ).images[0]
    
    output.save(os.path.join(opt.outfile, f"{time.time_ns()}.png"), 'PNG')
    print("CN: Image from Pose: done!")

if opt.mode == "PfI":
    generatePoseFromImage()
elif opt.mode == "IfP":
    generateImageFromPose()



