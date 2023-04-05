import os
import sys, time
import argparse
import json

from onnxruntime import SessionOptions

from diffusers import OnnxRuntimeModel
from transformers import CLIPTokenizer

from sd_xbackend import (
    GetPipe,
    GetSampler,
    ApplyArg,
    MakeImage
)

from modules.onnx.lora import blend_loras, buffer_external_data_tensors
ONNX_MODEL = "model.onnx"

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()
ApplyArg(parser)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()
prov = "DmlExecutionProvider"

pipe = GetPipe(opt.mdlpath, opt.mode, True, opt.nsfw, False)

def ApplyLora():
    blended_unet = blend_loras(opt.mdlpath + "/unet/" + ONNX_MODEL, opt.lora_path, "unet", opt.lora_strength)
    (unet_model, unet_data) = buffer_external_data_tensors(blended_unet)
    unet_names, unet_values = zip(*unet_data)
    sess = SessionOptions()
    sess.add_external_initializers(list(unet_names), list(unet_values))
    
    blended_te = blend_loras(opt.mdlpath + "/text_encoder/" + ONNX_MODEL, opt.lora_path, "text_encoder", opt.lora_strength)
    (te_model, te_data) = buffer_external_data_tensors(blended_te)
    te_names, te_values = zip(*te_data)
    sess_te = SessionOptions()
    sess_te.add_external_initializers(list(te_names), list(te_values))
    
    pipe.unet = OnnxRuntimeModel(OnnxRuntimeModel.load_model(unet_model.SerializeToString(), provider=prov, sess_options=sess))
    pipe.text_encoder = OnnxRuntimeModel(OnnxRuntimeModel.load_model(te_model.SerializeToString(), provider=prov, sess_options=sess_te))


if opt.lora:
    ApplyLora()

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
