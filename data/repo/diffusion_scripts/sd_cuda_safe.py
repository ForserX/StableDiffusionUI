import os
import sys, time
import argparse
import json
import torch

from diffusers import AutoencoderKL
from transformers import CLIPTokenizer




from sd_xbackend import (
    GetPipe,
    GetSampler,
    ApplyArg,
    MakeImage,
    ApplyLoRA,
    ApplyHyperNetwork
)

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()
ApplyArg(parser)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()
prov = "DmlExecutionProvider"

if opt.precision == "fp16":
    fptype = torch.float16
else:
    fptype = torch.float32

pipe = GetPipe(opt.mdlpath, opt.mode, False, opt.nsfw, opt.precision == "fp16")
pipe.to(opt.device)
safe_unet = pipe.unet
    
if opt.dlora:
    pipe.unet.load_attn_procs(opt.lora_path)


print("SD: Model preload: done")

while True:
    message = input()
    print(message)

    if not message:
        time.sleep(0.01)

    if message == "stop":
        break
    
    data = json.loads(message)
    
    if data['LoRA'] != old_lora_json:
        # Setup default unet
        pipe.unet = safe_unet

        for item in data['LoRA']:
            l_name = item['Name']
            l_alpha = item['Value']

            print(f"Apply {l_alpha} lora:{l_name}")
            ApplyLoRA(pipe, l_name, opt.device, opt.precision == "fp16", l_alpha)

        old_lora_json = data['LoRA']


    if not data['VAE'] == "Default":
        print("Load custom vae")
        pipe.vae = AutoencoderKL.from_pretrained(data['VAE'] + "/vae", torch_dtype=fptype)
        pipe.to(opt.device)
        
    eta = GetSampler(pipe, data['Sampler'], data['ETA'])
    print(f"Prompt: {data['Prompt']}")
    print(f"Neg rompt: {data['NegPrompt']}")
    
    seed = data['StartSeed']
    for i in range(data['TotalCount']):
        MakeImage(pipe, data['Mode'], eta, data['Prompt'], data['NegPrompt'], data['Steps'], data['Width'], data['Height'], seed, data['CFG'], data['ImgScale'], data['Device'] ,data['Image'] , data['ImgScale'], data['Mask'], data['WorkingDir'])
        seed = seed + 1
        
    print("SD Pipeline: Generating done!")
