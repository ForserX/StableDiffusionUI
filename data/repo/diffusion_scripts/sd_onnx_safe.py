import os
import sys, time
import argparse
import json

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
    
print("SD: Model preload: done")

while True:
    message = input()
    print(message)

    if not message:
        time.sleep(0.01)

    if message == "stop":
        break
    
    data = json.loads(message)

    if not data['VAE'] == "default":
        print("Load custom vae")
        pipe.vae_decoder = OnnxRuntimeModel.from_pretrained(data['VAE'] + "/vae_decoder", provider=prov)
        
 #   if local_args.inversion is not None:
 #       pipe.text_encoder = OnnxRuntimeModel.from_pretrained(opt.mdlpath + "/textual_inversion_merges/" + opt.inversion + "/text_encoder", provider="CPUExecutionProvider")
 #       pipe.tokenizer = CLIPTokenizer.from_pretrained(opt.mdlpath + "/textual_inversion_merges/" + opt.inversion + "/tokenizer")
    
    eta = GetSampler(pipe, data['Sampler'], data['ETA'])
    print(f"Prompt: {data['Prompt']}")
    print(f"Neg rompt: {data['NegPrompt']}")
    
    seed = data['StartSeed']
    for i in range(data['TotalCount']):
        MakeImage(pipe, data['Mode'], eta, data['Prompt'], data['NegPrompt'], data['Steps'], data['Width'], data['Height'], seed, data['CFG'], data['ImgScale'], "onnx", data['Image'] , data['ImgScale'], data['Mask'], data['WorkingDir'])
        seed = seed + 1
        
    print("SD Pipeline: Generating done!")
