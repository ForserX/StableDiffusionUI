import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="Path to model checkpoint file",
    default="timbrooks/instruct-pix2pix",
    dest='model',
)

parser.add_argument(
    "-img",
    "--img",
    type=str,
    default="",
    help="Specify generation mode",
    dest='img',
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

opt = parser.parse_args()

# TODO: Add CPU
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(opt.model, torch_dtype=torch.float16).to("cuda")

def download_image():
    image = PIL.Image.open(opt.img)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


image = download_image(opt.img)

edit = pipe(opt.prompt, negative_prompt=opt.prompt_neg , image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images[0]
images[0].save("snowy_mountains.png")