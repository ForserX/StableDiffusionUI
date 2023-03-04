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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_embeds = torch.load(inversion, map_location=device)
    
    string_to_token = loaded_embeds["string_to_token"]
    string_to_param = loaded_embeds["string_to_param"]
    
    # separate token and embeds
    trained_token = list(string_to_token.keys())[0]
    embeds = string_to_param[trained_token]
    
    num_tokens = embeds.shape[0]
    print(f"generating %s layer tokens {num_tokens}")
    token = [f"{name}-{i}" for i in range(num_tokens)]

    print(f"found embedding for token {trained_token}, {embeds.shape}")

    tokenizer = CLIPTokenizer.from_pretrained(
        base_model,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        base_model,
        subfolder="text_encoder",
    )

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer."
        )

    print(f"added {num_added_tokens} tokens")

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    if len(embeds.shape) == 2:
        # multiple vectors in embeds
        for i in range(embeds.shape[0]):
            layer_embeds = embeds[i]
            layer_token = token[i]
            print(
                f"embedding {layer_embeds.shape} vector for layer {layer_token}"
            )
            token_id = tokenizer.convert_tokens_to_ids(layer_token)
            text_encoder.get_input_embeddings().weight.data[token_id] = layer_embeds
    else:
        # get the id for the token and assign the embeds
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    # conversion stuff
    text_input = tokenizer(
        "A sample prompt",
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