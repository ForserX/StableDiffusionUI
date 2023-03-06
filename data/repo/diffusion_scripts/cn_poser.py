from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler


import torch
import cv2
from PIL import Image
import numpy as np


from controlnet_aux import OpenposeDetector
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode", type=str, help="IfP - image from pose or PfI - pose from image", dest='mode',
)

parser.add_argument(
    "--model", type=str, help="Path to model checkpoint file", dest='model',
)
parser.add_argument(
    "--img", type=str, default="", help="Specify generation mode", dest='img',
)

parser.add_argument(
    "--outfile", type=str, default="", help="Specify generation mode", dest='outfile',
)

parser.add_argument(
    "--device", type=str, default="cpu", help="Choosen Device", dest='device',
)

parser.add_argument(
    "--precision", type=str, default="fp16", help="Pressicion fp16 or fp32", dest='precision',
)

parser.add_argument(
    "--seed", type=str, default="-1", help="seed", dest='seed',
)

parser.add_argument(
    "--pose", type=str, default="", help="input pose image", dest='pose',
)

parser.add_argument(
    "--steps", type=str, default="20", help="sheduller steps", dest='steps',
)

parser.add_argument(
    "--prompt", type=str, default="", help="prompt", dest='prompt',
)

parser.add_argument(
    "--neg_prompt", type=str, default="", help="negative prompt", dest='neg_prompt',
)


opt = parser.parse_args()

   
    


def generatePoseFromImage():
    print("processing generatePoseFromImage()")
    model = OpenposeDetector.from_pretrained(opt.model)
    in_img = Image.open(opt.img)
    
    img = model(in_img)
    img.save(opt.outfile)
    print(f"CN: Pose - {opt.outfile}")

def generateImageFromPose ():
    print("processing generateImageFromPose()")
    image = load_image( opt.pose )

    
    if opt.precision == "fp32":
        dtype = torch.float32
    else:
        dtype = torch.float16


    controlnet = ControlNetModel.from_pretrained(
        opt.model, torch_dtype=dtype
    )    
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model = opt.model,
        controlnet=controlnet,
        torch_dtype=dtype
    )

    if opt.device == "cpu":
        pipe.enable_model_cpu_offload()

    generator = torch.Generator(device=opt.device).manual_seed(opt.seed)
        
    output = pipe(
        prompt = opt.prompt,
        poses = opt.pose,
        negative_prompt = opt.neg_prompt,
        generator = generator,
        num_inference_steps=opt.steps
    )

    img = pipe.model()
    img.save(opt.outfile)
    print(f"CN: Pose - {opt.outfile}")





if opt.mode == "PfI":
    generatePoseFromImage()
elif opt.mode == "IfP":
    generateImageFromPose()



