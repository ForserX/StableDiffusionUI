from os import makedirs, path
from typing import List, Optional, Dict

from pathlib import Path

import numpy as np
import torch, safetensors
from transformers import CLIPTokenizer

ONNX_MODEL = "model.onnx"

def load_tensor(name: str, map_location=None) -> Optional[Dict]:
	_, extension = path.splitext(name)
	extension = extension[1:].lower()

	checkpoint = None
	if extension == "":
		# if no extension was intentional, do not search for others
		if path.exists(name):
			checkpoint = torch.load(name, map_location=map_location)
		else:
			for next_extension in ["safetensors", "ckpt", "pt", "bin"]:
				next_name = f"{name}.{next_extension}"
				if path.exists(next_name):
					checkpoint = load_tensor(next_name, map_location=map_location)
					if checkpoint is not None:
						break

	elif extension == "safetensors":
			checkpoint = safetensors.torch.load_file(name, device="cpu")
	elif extension in ["bin", "ckpt", "pt"]:
			checkpoint = torch.load(name, map_location=map_location)
	elif extension in ["onnx", "pt"]:
			checkpoint = torch.load(name, map_location=map_location)
	else:
			checkpoint = torch.load(name, map_location=map_location)

	if checkpoint is not None and "state_dict" in checkpoint:
		checkpoint = checkpoint["state_dict"]

	return checkpoint

@torch.no_grad()
def blend_textual_inversions(
	text_encoder,
	tokenizer: CLIPTokenizer,
	inversions: str,
	inversions_alpha: float
):
	# always load to CPU for blending
	device = torch.device("cpu")
	dtype = np.float32
	embeds = {}

	inversion_format = None
	base_token = Path(inversions).stem
	tokenList = []
	loaded_embeds = load_tensor(inversions, map_location=device)

	if inversion_format is None:
		keys: List[str] = list(loaded_embeds.keys())
		if len(keys) == 1 and keys[0].startswith("<") and keys[0].endswith(">"):
			inversion_format = "concept"
		elif "emb_params" in keys:
			inversion_format = "params"
		elif "string_to_token" in keys and "string_to_param" in keys:
			inversion_format = "embeddings"
		else:
			print(
				f"unknown Textual Inversion format, no recognized keys: %s", keys
			)
			return

	if inversion_format == "concept":
		# separate token and the embeds
		token = list(loaded_embeds.keys())[0]

		layer = loaded_embeds[token].numpy().astype(dtype)
		layer *= inversions_alpha

		if base_token in embeds:
			embeds[base_token] += layer
		else:
			embeds[base_token] = layer

		if token in embeds:
			embeds[token] += layer
		else:
			embeds[token] = layer

	elif inversion_format == "embeddings":
		string_to_token = loaded_embeds["string_to_token"]
		string_to_param = loaded_embeds["string_to_param"]

		# separate token and embeds
		token = list(string_to_token.keys())[0]
		trained_embeds = string_to_param[token]

		num_tokens = trained_embeds.shape[0]
		sum_layer = np.zeros(trained_embeds[0, :].shape)

		
		for i in range(num_tokens):
			token = f"{base_token}-{i}"
			tokenList.append(token)
			layer = trained_embeds[i, :].numpy().astype(dtype)
			layer *= inversions_alpha

			sum_layer += layer
			if token in embeds:
				embeds[token] += layer
			else:
				embeds[token] = layer

		# add base and sum tokens to embeds
		if base_token in embeds:
			embeds[base_token] += sum_layer
		else:
			embeds[base_token] = sum_layer

		sum_token = f"{base_token}-all"
		if sum_token in embeds:
			embeds[sum_token] += sum_layer
		else:
			embeds[sum_token] = sum_layer

	elif inversion_format == "params":
		string_to_param = loaded_embeds["emb_params"]

		num_tokens = string_to_param.shape[0]
		sum_layer = np.zeros(string_to_param[0, :].shape)

		for i in range(num_tokens):
			token = f"{base_token}-{i}"
			tokenList.append(token)
			layer = string_to_param[i, :].numpy().astype(dtype)
			layer *= inversions_alpha

			sum_layer += layer
			if token in embeds:
				embeds[token] += layer
			else:
				embeds[token] = layer

		# add base and sum tokens to embeds
		if base_token in embeds:
			embeds[base_token] += sum_layer
		else:
			embeds[base_token] = sum_layer

		sum_token = f"{base_token}-all"
		if sum_token in embeds:
			embeds[sum_token] += sum_layer
		else:
			embeds[sum_token] = sum_layer
	else:
		raise ValueError(f"unknown Textual Inversion format: {inversion_format}")

	# add the tokens to the tokenizer

	num_added_tokens = tokenizer.add_tokens(list(embeds.keys()))
	if num_added_tokens == 0:
		raise ValueError(
			f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer."
		)
	
	prompt_tokens = "("
	for i in range(num_added_tokens):
		prompt_tokens = prompt_tokens + f"{base_token}-{i}, "

	prompt_tokens = prompt_tokens + "), "

	print(f"added {num_added_tokens} tokens from {base_token} textual inversion")

	# resize the token embeddings
	text_encoder.resize_token_embeddings(len(tokenizer))
	

	if len(trained_embeds.shape) == 2:
		# multiple vectors in embeds
		for i in range(trained_embeds.shape[0]):
			layer_embeds = trained_embeds[i]
			layer_token = tokenList[i]
			#print(
			#	f"embedding {layer_embeds.shape} vector for layer {layer_token}"
			#)
			token_id = tokenizer.convert_tokens_to_ids(layer_token)
			text_encoder.get_input_embeddings().weight.data[token_id] = layer_embeds
	else:
		# get the id for the token and assign the embeds
		token_id = tokenizer.convert_tokens_to_ids(token)
		text_encoder.get_input_embeddings().weight.data[token_id] = embeds

	return (tokenizer, text_encoder, prompt_tokens)

