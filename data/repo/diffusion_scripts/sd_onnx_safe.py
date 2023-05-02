import os, sys, time, argparse, json, torch, onnx

from diffusers import OnnxRuntimeModel
from sd_xbackend import Device, ONNX_MODEL
from transformers import CLIPTokenizer

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()
Device.ApplyArg(parser)
opt = parser.parse_args()

pipe = None
PipeDevice = Device("onnx", torch.float32)

pipe = PipeDevice.GetPipe(opt.mdlpath, opt.mode, opt.nsfw)
safe_unet = pipe.unet

print("SD: Model preload: done")

old_lora_json = None
old_te_json = None
old_ten_json = None
old_vae_json = "Default"

onnx_te_model = onnx.load(opt.mdlpath + "/text_encoder/" + ONNX_MODEL)

prompt_tokens = ""
prompt_neg_tokens = ""

tokenizer_extract = False

while True:
    message = input()

    if not message:
        time.sleep(0.01)

    if message == "stop":
        break
    
    data = json.loads(message)
    tokenizer_extract = False

    if (data['LoRA'] != old_lora_json) or (data['TI'] != old_te_json) or (data['TINeg'] != old_ten_json):
        onnx_te_model = onnx.load(opt.mdlpath + "/text_encoder/" + ONNX_MODEL)
        
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
            
            onnx_te_model, new_pt = PipeDevice.ApplyTE(onnx_te_model, l_name, l_alpha, pipe)
            tokenizer_extract = True
            prompt_tokens = prompt_tokens + new_pt
        
        for item in data['TINeg']:
            l_name = item['Name']
            l_alpha = item['Value']
            
            onnx_te_model, new_pt = PipeDevice.ApplyTE(onnx_te_model, l_name, l_alpha, pipe)
            tokenizer_extract = True
            prompt_neg_tokens = prompt_neg_tokens + new_pt
        
        old_te_json = data['TI']
        old_ten_json = data['TINeg']
        
    if data['LoRA'] != old_lora_json:
        # Setup default unet
        pipe.unet = safe_unet

        for item in data['LoRA']:
            l_name = item['Name']
            l_alpha = item['Value']

            print(f"Apply {l_alpha} lora:{l_name}")
            onnx_te_model = PipeDevice.ApplyLoraONNX(opt, onnx_te_model, l_name, l_alpha, pipe)
            tokenizer_extract = True
        
        old_lora_json = data['LoRA']
        
    if tokenizer_extract:
        # Convert te model to Runtime
        pipe.text_encoder = PipeDevice.ONNXProto2Runtime(onnx_te_model)

    if (data['VAE'] != "Default") or (data['VAE'] != old_vae_json):
        print("Load custom vae")
        pipe.vae_decoder = OnnxRuntimeModel.from_pretrained(data['VAE'] + "/vae_decoder", provider=PipeDevice.prov)
        old_vae_json = data['VAE']
        
    eta = PipeDevice.GetSampler(pipe, data['Sampler'], data['ETA'])
    print(f"Prompt: {data['Prompt']}")
    print(f"Neg prompt: {data['NegPrompt']}")

    seed = data['StartSeed']
    for i in range(data['TotalCount']):
        PipeDevice.MakeImage(pipe, data['Mode'], eta, prompt_tokens + data['Prompt'], prompt_neg_tokens + data['NegPrompt'], data['Steps'], data['Width'], data['Height'], seed, data['CFG'], data['ImgScale'], data['Image'] , data['ImgScale'], data['Mask'], data['WorkingDir'], data['BatchSize'])
        seed = seed + 1
        
    print("SD Pipeline: Generating done!")
