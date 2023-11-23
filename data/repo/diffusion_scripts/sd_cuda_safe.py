import os, sys, time
import argparse, json, torch

from diffusers import AutoencoderKL
from transformers import CLIPTokenizer

os.chdir(sys.path[0])
from sd_xbackend import Device


parser = argparse.ArgumentParser()
Device.ApplyArg(parser)

opt = parser.parse_args()

if opt.precision == "fp16":
    fptype = torch.float16
else:
    fptype = torch.float32

PipeDevice = Device(opt.device, fptype)

pipe = PipeDevice.GetPipe(opt.mdlpath, opt.mode, opt.nsfw, "")
pipe.to(PipeDevice.device, fptype)
    
if opt.dlora:
    pipe.unet.load_attn_procs(opt.lora_path)

print("SD: Model preload: done")

old_lora_json = None
old_te_json = None
old_ten_json = None
old_vae_json = "Default"

prompt_tokens = ""
prompt_neg_tokens = ""

while True:
    message = input()
    print(message)

    if not message:
        time.sleep(0.01)

    if message == "stop":
        break
    
    data = json.loads(message)
    
    if (data['LoRA'] != old_lora_json) or (data['TI'] != old_te_json) or (data['TINeg'] != old_ten_json):
        pipe.unet = pipe.unet.from_pretrained(opt.mdlpath+"/unet")
        pipe.text_encoder = pipe.text_encoder.from_pretrained(opt.mdlpath+"/text_encoder")
        pipe.to(PipeDevice.device, PipeDevice.fptype)
        
        # Hard reload
        old_lora_json = None
        old_te_json = None
        old_ten_json = None
    
    if (data['TI'] != old_te_json) or (data['TINeg'] != old_ten_json):
        prompt_tokens = ""
        prompt_neg_tokens = ""

        # Load base tokenizer
        pipe.tokenizer = CLIPTokenizer.from_pretrained(opt.mdlpath + "/tokenizer")

        for item in data['TI']:
            l_name = item['Name']
            l_alpha = item['Value']
            
            pipe, new_pt = PipeDevice.ApplyTI2CUDA(l_name, l_alpha, pipe)
            tokenizer_extract = True
            prompt_tokens = prompt_tokens + new_pt
        
        for item in data['TINeg']:
            l_name = item['Name']
            l_alpha = item['Value']
            
            pipe, new_pt = PipeDevice.ApplyTI2CUDA(l_name, l_alpha, pipe)
            tokenizer_extract = True
            prompt_neg_tokens = prompt_neg_tokens + new_pt
        
        old_te_json = data['TI']
        old_ten_json = data['TINeg']
        
    if data['LoRA'] != old_lora_json:
        # Setup default unet

        for item in data['LoRA']:
            l_name = item['Name']
            l_alpha = item['Value']            

            print(f"Apply {l_alpha} lora:{l_name}")
            PipeDevice.ApplyLoRA(pipe.unet, pipe.text_encoder, l_name, l_alpha)

        old_lora_json = data['LoRA']

    if data['VAE'] != old_vae_json:
        print("Load custom vae")
        pipe.vae = AutoencoderKL.from_pretrained(data['VAE'] + "/vae", torch_dtype=PipeDevice.fptype).to(PipeDevice.device)
        old_vae_json = data['VAE']

    eta = PipeDevice.GetSampler(pipe, data['Sampler'], data['ETA'])
    print(f"Prompt: {data['Prompt']}")
    print(f"Neg rompt: {data['NegPrompt']}")
    
    seed = data['StartSeed']
    for i in range(data['TotalCount']):
        PipeDevice.MakeImage(pipe, data['Mode'], eta, data['Prompt'] + prompt_tokens, data['NegPrompt'] + prompt_neg_tokens, data['Steps'], data['Width'], data['Height'], seed, data['CFG'], data['ImgScale'], data['Image'] , data['ImgScale'], data['Mask'], data['WorkingDir'], data['BatchSize'])
        seed = seed + 1
        
    print("SD Pipeline: Generating done!")
