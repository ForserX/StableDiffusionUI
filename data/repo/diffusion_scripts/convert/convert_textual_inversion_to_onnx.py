from os import makedirs, path

import argparse
import torch

from torch.onnx import export
from transformers import CLIPTextModel, CLIPTokenizer

@torch.no_grad()
def convert_diffusion_textual_inversion(
    out_name: str, name: str, base_model: str, inversion: str,
):
    dest_path = path.join(out_name, f"{name}")
    print(
        "converting Textual Inversion: %s + %s -> %s", base_model, inversion, dest_path
    )

    makedirs(path.join(dest_path, "text_encoder"), exist_ok=True)
    
    tokenizer = CLIPTokenizer.from_pretrained(
        base_model,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        base_model,
        subfolder="text_encoder",
    )
    
    loaded_embeds = torch.load(inversion)

    trained_token = name
    if len(loaded_embeds) > 2:
        embeds = loaded_embeds["string_to_param"]["*"][-1, :]
    else:
        # separate token and the embeds
        trained_token = list(loaded_embeds.keys())[0]
        embeds = loaded_embeds[trained_token]

    token = trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    print(f"Initial token: {trained_token} - {num_added_tokens}")
    i = 1

    while num_added_tokens == 0:
        print(f"The tokenizer already contains the token {token}.")
        token = f"{token[:-1]}-{i}>"
        print(f"Attempting to add the token {token}.")
        num_added_tokens = tokenizer.add_tokens(token)
        i += 1

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    # conversion stuff
    text_input = tokenizer(
        trained_token,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    print("saving tokenizer for textual inversion")
    tokenizer.save_pretrained(path.join(dest_path, "tokenizer"))

    print("saving text encoder for textual inversion")
    export(
        text_encoder,
        (text_input.input_ids.to(dtype=torch.int32)),
        f=path.join(dest_path, "text_encoder", "model.onnx"),
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
        },
        do_constant_folding=True,
        opset_version=14,
    )

    print("textual inversion saved to %s", dest_path)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Path to model checkpoint file",
    dest='model',
)

parser.add_argument(
    "--textinv",
    type=str,
    help="Path to model checkpoint file",
    dest='textinv',
)

parser.add_argument(
    "--name",
    type=str,
    help="Path to model checkpoint file",
    dest='name',
)

parser.add_argument(
    "--dest",
    type=str,
    help="Path to model checkpoint file",
    dest='dest',
)

arg = parser.parse_args()

convert_diffusion_textual_inversion(arg.dest, arg.name, arg.model, arg.textinv)
print("SD: TI Done!")