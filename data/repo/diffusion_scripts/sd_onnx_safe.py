import os
import sys, time
import argparse
import json

from diffusers import OnnxRuntimeModel

from sd_xbackend import (
    GetPipe,
    ApplyLoraONNX,
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
pipe = None

pipe = GetPipe(opt.mdlpath, opt.mode, True, opt.nsfw, False)
safe_unet = pipe.unet

print("SD: Model preload: done")

old_lora_json = None

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
            ApplyLoraONNX(opt, l_name, l_alpha, pipe)
        
        old_lora_json = data['LoRA']

    if not data['VAE'] == "Default":
        print("Load custom vae")
        pipe.vae_decoder = OnnxRuntimeModel.from_pretrained(data['VAE'] + "/vae_decoder", provider=prov)
        
    eta = GetSampler(pipe, data['Sampler'], data['ETA'])
    print(f"Prompt: {data['Prompt']}")
    print(f"Neg rompt: {data['NegPrompt']}")
    
    seed = data['StartSeed']
    for i in range(data['TotalCount']):
        MakeImage(pipe, data['Mode'], eta, data['Prompt'], data['NegPrompt'], data['Steps'], data['Width'], data['Height'], seed, data['CFG'], data['ImgScale'], "onnx", data['Image'] , data['ImgScale'], data['Mask'], data['WorkingDir'])
        seed = seed + 1
        
    print("SD Pipeline: Generating done!")
