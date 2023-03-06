import os
import sys
import torch

from diffusers import AutoencoderKL

from sd_xbackend import (
    GetPipe,
    GetSampler,
    ApplyArg,
    MakeImage,
    ApplyLoRA
)

import argparse

os.chdir(sys.path[0])

parser = argparse.ArgumentParser()
ApplyArg(parser)

if len(sys.argv)==1:
    parser.print_help()
    parser.exit()

opt = parser.parse_args()

if opt.precision == "fp16":
    fptype = torch.float16
else:
    fptype = torch.float32


pipe = GetPipe(opt.mdlpath, opt.mode, False, opt.nsfw, opt.precision == "fp16")

if not opt.vae == "default":
    pipe.vae = AutoencoderKL.from_pretrained(opt.vae + "/vae", torch_dtype=fptype)
    
pipe.to(opt.device)
eta = GetSampler(pipe, opt.scmode, opt.eta)

if opt.inversion is not None:
    loaded_embeds = torch.load(opt.inversion, opt.device)
    from pathlib import Path

    trained_token = Path(opt.inversion).stem
    if len(loaded_embeds) > 2:
        embeds = loaded_embeds["string_to_param"]["*"][-1, :]
    else:
        # separate token and the embeds
        trained_token = list(loaded_embeds.keys())[0]
        embeds = loaded_embeds[trained_token]

    token = trained_token
    num_added_tokens = pipe.tokenizer.add_tokens(token)
    print(f"Initial token: {trained_token} - {num_added_tokens}")
    i = 1
    while num_added_tokens == 0:
        print(f"The tokenizer already contains the token {token}.")
        token = f"{token[:-1]}-{i}>"
        print(f"Attempting to add the token {token}.")
        num_added_tokens = pipe.tokenizer.add_tokens(token)
        i += 1

    # resize the token embeddings
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

    # get the id for the token and assign the embeds
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    pipe.text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    
# LoRA magic
if opt.lora:
    ApplyLoRA(pipe, opt.lora_path, opt.device, opt.precision == "fp16")

print("SD: Model loaded")
print(f"Prompt: {opt.prompt}")
print(f"Neg rompt: {opt.prompt_neg}")

for i in range(opt.totalcount):
    MakeImage(pipe, opt.mode, eta, opt.prompt, opt.prompt_neg, opt.steps, opt.width, opt.height, opt.seed, opt.guidance_scale, opt.img , opt.imgscale, opt.imgmask, opt.outpath)
    opt.seed = opt.seed + 1
    
print("SD: Generating done!")
