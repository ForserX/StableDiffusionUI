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
    
# LoRA magic
if opt.lora:
    ApplyLoRA(pipe, opt.lora_path, opt.device, opt.precision == "fp16", 0.75)

if opt.dlora:
    pipe.unet.load_attn_procs(opt.lora_path)


ApplyHyperNetwork(pipe, "D:/Drive/NeuralNetworks/StableDiffusionUI-Shark-AMD/data/models/hypernetwork/samdoesartsHyper.pt" ,"cuda", opt.precision == "fp16",0.75)

print("SD: Model preload: done")

while True:
    message = input()
    print(message)

    if not message:
        time.sleep(0.01)

    if message == "stop":
        break
    
    data = json.loads(message)
    

    if not data['VAE'] == "Default":
        print("Load custom vae")
        pipe.vae = AutoencoderKL.from_pretrained(data['VAE'] + "/vae", torch_dtype=fptype)
        pipe.to(opt.device)

 #   if local_args.inversion is not None:
 #       pipe.text_encoder = OnnxRuntimeModel.from_pretrained(opt.mdlpath + "/textual_inversion_merges/" + opt.inversion + "/text_encoder", provider="CPUExecutionProvider")
 #       pipe.tokenizer = CLIPTokenizer.from_pretrained(opt.mdlpath + "/textual_inversion_merges/" + opt.inversion + "/tokenizer")
    
    eta = GetSampler(pipe, data['Sampler'], data['ETA'])
    print(f"Prompt: {data['Prompt']}")
    print(f"Neg rompt: {data['NegPrompt']}")
    
    seed = data['StartSeed']
    for i in range(data['TotalCount']):
        MakeImage(pipe, data['Mode'], eta, data['Prompt'], data['NegPrompt'], data['Steps'], data['Width'], data['Height'], seed, data['CFG'], data['ImgScale'], data['Device'] ,data['Image'] , data['ImgScale'], data['Mask'], data['WorkingDir'])
        seed = seed + 1
        
    print("SD Pipeline: Generating done!")
