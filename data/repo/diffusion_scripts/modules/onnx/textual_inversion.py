from os import makedirs, path
from typing import List, Optional, Tuple, Dict

from pathlib import Path

import numpy as np
import torch, safetensors, onnx
from onnx import ModelProto, numpy_helper
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
) -> Tuple[ModelProto, CLIPTokenizer]:
	# always load to CPU for blending
	device = torch.device("cpu")
	dtype = np.float32
	embeds = {}

	inversion_format = None
	base_token = Path(inversions).stem
	
	loaded_embeds = load_tensor(inversions, map_location=device)

	if inversion_format is None:
		keys: List[str] = list(loaded_embeds.keys())
		if len(keys) == 1 and keys[0].startswith("<") and keys[0].endswith(">"):
			inversion_format = "concept"
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
	# text_encoder.resize_token_embeddings(len(tokenizer))
	embedding_node = [
		n
		for n in text_encoder.graph.initializer
		if n.name == "text_model.embeddings.token_embedding.weight"
	][0]
	base_weights = numpy_helper.to_array(embedding_node)

	weights_dim = base_weights.shape[1]
	zero_weights = np.zeros((num_added_tokens, weights_dim))
	embedding_weights = np.concatenate((base_weights, zero_weights), axis=0)

	for token, weights in embeds.items():
		token_id = tokenizer.convert_tokens_to_ids(token)
		embedding_weights[token_id] = weights

	# replace embedding_node
	for i in range(len(text_encoder.graph.initializer)):
		if (
			text_encoder.graph.initializer[i].name
			== "text_model.embeddings.token_embedding.weight"
		):
			new_initializer = numpy_helper.from_array(
				embedding_weights.astype(base_weights.dtype), embedding_node.name
			)
			del text_encoder.graph.initializer[i]
			text_encoder.graph.initializer.insert(i, new_initializer)

	return (tokenizer, prompt_tokens)

