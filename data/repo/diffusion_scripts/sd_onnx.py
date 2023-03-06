import os
import sys
import argparse

from diffusers import OnnxRuntimeModel
from transformers import CLIPTokenizer

from sd_xbackend import (
    GetPipe,
    GetSampler,
    ApplyArg,
    MakeImage
)

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()
ApplyArg(parser)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()
prov = "DmlExecutionProvider"

pipe = GetPipe(opt.mdlpath, opt.mode, True, opt.nsfw, False)

if not opt.vae == "default":
    print("Load custom vae")
    pipe.vae_decoder = OnnxRuntimeModel.from_pretrained(opt.vae + "/vae_decoder", provider=prov)
    
if opt.inversion is not None:
    pipe.text_encoder = OnnxRuntimeModel.from_pretrained(opt.mdlpath + "/textual_inversion_merges/" + opt.inversion + "/text_encoder", provider="CPUExecutionProvider")
    pipe.tokenizer = CLIPTokenizer.from_pretrained(opt.mdlpath + "/textual_inversion_merges/" + opt.inversion + "/tokenizer")

eta = GetSampler(pipe, opt.scmode, opt.eta)

print("SD: Model loaded")
print(f"Prompt: {opt.prompt}")
print(f"Neg rompt: {opt.prompt_neg}")

for i in range(opt.totalcount):
    MakeImage(pipe, opt.mode, eta, opt.prompt, opt.prompt_neg, opt.steps, opt.width, opt.height, opt.seed, opt.guidance_scale, opt.img , opt.imgscale, opt.imgmask, opt.outpath)
    opt.seed = opt.seed + 1
    
print("SD: Generating done!")
