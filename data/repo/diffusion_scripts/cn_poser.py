from diffusers import UniPCMultistepScheduler, AutoencoderKL, OnnxRuntimeModel
import torch, os, time, numpy

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

prov = "DmlExecutionProvider"

if opt.precision == "fp16":
    fptype = torch.float16
else:
    fptype = torch.float32


def generatePoseFromImage():
    print("processing generatePoseFromImage()")
    model = OpenposeDetector.from_pretrained(opt.mdlpath)
    in_img = Image.open(opt.img)
    
    img = model(in_img)
    img.save(opt.outfile)
    print(f"CN: Pose - {opt.outfile}")

pipe = None
image = None

def generateImageFromPose ():
    print("processing generateImageFromPose()")

    if opt.mode == "IfPONNX":
        generator = numpy.random.seed(opt.seed)
    else:
        generator = torch.Generator(device=opt.device).manual_seed(opt.seed)
        
    output = pipe(
        opt.prompt,
        image,
        opt.height,
        opt.width, 
        negative_prompt = opt.prompt_neg,
        generator = generator,
        num_inference_steps=opt.steps
    ).images[0]
    
    output.save(os.path.join(opt.outfile, f"{time.time_ns()}.png"), 'PNG')
    print("CN: Image from Pose: done!")

if opt.mode == "PfI":
    generatePoseFromImage()

elif opt.mode == "IfPONNX":
    print("CN: ONNX initial")
    image = Image.open(opt.pose)
    image.convert("RGB").resize((opt.width, opt.height))

    pipe = GetPipeCN(opt.mdlpath, opt.cn_model, opt.nsfw, opt.precision == "fp16", True)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if not opt.vae == "default":
        pipe.vae_decoder = OnnxRuntimeModel.from_pretrained(opt.vae + "/vae_decoder", provider=prov)
        
    print("CN: Model loaded")
    for i in range(opt.totalcount):
        generateImageFromPose()
        opt.seed = opt.seed + 1

elif opt.mode == "IfP":
    image = Image.open(opt.pose)
    image.convert("RGB").resize((opt.width, opt.height))

    pipe = GetPipeCN(opt.mdlpath, opt.cn_model, opt.nsfw, opt.precision == "fp16")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if not opt.vae == "default":
        pipe.vae = AutoencoderKL.from_pretrained(opt.vae + "/vae", torch_dtype=fptype)
        
    pipe.to(opt.device)

    #if opt.device == "cpu":
    #    pipe.enable_model_cpu_offload()

    # LoRA magic
    if opt.lora:
        ApplyLoRA(pipe, opt.lora_path, opt.device, opt.precision == "fp16", 0.75)

    print("CN: Model loaded")
    for i in range(opt.totalcount):
        generateImageFromPose()
        opt.seed = opt.seed + 1
