# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, os, shutil, json
from pathlib import Path

import torch
from torch.onnx import export

import onnx
import onnxruntime as ort

from olive.model import ONNXModel
from olive.workflows import run as olive_run

from diffusers.models.attention_processor import AttnProcessor
from diffusers import StableDiffusionPipeline, OnnxStableDiffusionPipeline, OnnxRuntimeModel
from unet_2d_condition_cnet import UNet2DConditionModel_Cnet

@torch.no_grad()
def convert_to_fp16(
	model_path
):
	'''Converts an ONNX model on disk to FP16'''
	from onnxruntime.transformers.float16 import convert_float_to_float16

	model_dir=os.path.dirname(model_path)
	# Breaking down in steps due to Windows bug in convert_float_to_float16_model_path
	onnx.shape_inference.infer_shapes_path(model_path)
	fp16_model = onnx.load(model_path)
	fp16_model = convert_float_to_float16(
		fp16_model, keep_io_types=True, disable_shape_infer=True
	)
	# clean up existing tensor files
	shutil.rmtree(model_dir)
	os.mkdir(model_dir)
	# save FP16 model
	onnx.save(fp16_model, model_path)

def onnx_export(
	model,
	model_args: tuple,
	output_path: Path,
	ordered_input_names,
	output_names,
	dynamic_axes,
	opset,
):
	output_path.parent.mkdir(parents=True, exist_ok=True)
	export(
		model,
		model_args,
		f=output_path.as_posix(),
		input_names=ordered_input_names,
		output_names=output_names,
		dynamic_axes=dynamic_axes,
		do_constant_folding=True,
		opset_version=opset,
	)

def olive_export(model_path : str, output_path: str, submodel : str):
	fin= open(f"./model_data/olive/config_{submodel}.json", "r")
	olive_config = json.load(fin)
	
	olive_config["input_model"]["config"]["model_path"] = model_path
	olive_run(olive_config)

	footprints_file_path = (f"./footprints/{submodel}_gpu-dml_footprints.json")

	footprint_file = open(footprints_file_path, "r")
	footprints = json.load(footprint_file)
	
	optimizer_footprint = None
	conversion_footprint = None
	
	for _, footprint in footprints.items():
		if footprint["from_pass"] == "OrtTransformersOptimization":
			optimizer_footprint = footprint
	
	optimized_olive_model = ONNXModel(**optimizer_footprint["model_config"]["config"])
	print(f"Optimized Model   : {optimized_olive_model.model_path}")

	outfile_path = str(output_path) + f"/{submodel}/"

	if not os.path.exists(outfile_path):
		os.makedirs(outfile_path)
		
	if os.path.exists(outfile_path + "model.onnx"):
		os.remove(outfile_path + "model.onnx")

	shutil.move(optimized_olive_model.model_path, outfile_path + "model.onnx")
	
	footprints = None
	footprint_file = None
	olive_config = None

	if os.path.exists("./cache"):
		shutil.rmtree("./cache")

	if os.path.exists("./footprints"):
		shutil.rmtree("./footprints")


@torch.no_grad()
def convert_models(model_path: str, output_path: str, opset: int, fp16: bool = False):
	dtype = torch.float16 if fp16 else torch.float32
	if fp16 and torch.cuda.is_available():
		device = "cuda"
	elif fp16 and not torch.cuda.is_available():
		raise ValueError("`float16` model export is only supported on GPUs with CUDA")
	else:
		device = "cpu"
	pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
	output_path = Path(output_path)

	# TEXT ENCODER
	num_tokens = pipeline.text_encoder.config.max_position_embeddings
	text_hidden_size = pipeline.text_encoder.config.hidden_size
	olive_export(model_path, output_path, "text_encoder")
	
	# UNET
	pipeline.unet.set_attn_processor(AttnProcessor())
	
	unet_in_channels = pipeline.unet.config.in_channels
	unet_sample_size = pipeline.unet.config.sample_size
	unet_path = output_path / "unet" / "model.onnx"
	onnx_export(
		pipeline.unet,
		model_args=(
			torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
			torch.randn(2).to(device=device, dtype=dtype),
			torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
			False,
		),
		output_path=unet_path,
		ordered_input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
		output_names=["out_sample"],  # has to be different from "sample" for correct tracing
		dynamic_axes={
			"sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
			"timestep": {0: "batch"},
			"encoder_hidden_states": {0: "batch", 1: "sequence"},
		},
		opset=opset,
	)
	unet_model_path = str(unet_path.absolute().as_posix())
	unet_dir = os.path.dirname(unet_model_path)
	unet = onnx.load(unet_model_path)
	# clean up existing tensor files
	shutil.rmtree(unet_dir)
	os.mkdir(unet_dir)
	# collate external tensor files into one
	onnx.save_model(
		unet,
		unet_model_path,
		save_as_external_data=True,
		all_tensors_to_one_file=True,
		location="weights.pb",
		convert_attribute=False,
	)
	del pipeline.unet
	convert_to_fp16(unet_model_path)
	
	# UNET CONTROLNET
	pipe_cnet = UNet2DConditionModel_Cnet.from_pretrained(model_path, subfolder = "unet")
	
	pipe_cnet.set_attn_processor(AttnProcessor())
	
	cnet_path = output_path / "cnet" / "model.onnx"
	onnx_export(
		pipe_cnet,
		model_args=(
			torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
			torch.randn(2).to(device=device, dtype=dtype),
			torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
			torch.randn(2, 320, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
			torch.randn(2, 320, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
			torch.randn(2, 320, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
			torch.randn(2, 320, unet_sample_size//2,unet_sample_size//2).to(device=device, dtype=dtype),
			torch.randn(2, 640, unet_sample_size//2,unet_sample_size//2).to(device=device, dtype=dtype),
			torch.randn(2, 640, unet_sample_size//2,unet_sample_size//2).to(device=device, dtype=dtype),
			torch.randn(2, 640, unet_sample_size//4,unet_sample_size//4).to(device=device, dtype=dtype),
			torch.randn(2, 1280, unet_sample_size//4,unet_sample_size//4).to(device=device, dtype=dtype),
			torch.randn(2, 1280, unet_sample_size//4,unet_sample_size//4).to(device=device, dtype=dtype),
			torch.randn(2, 1280, unet_sample_size//8,unet_sample_size//8).to(device=device, dtype=dtype),
			torch.randn(2, 1280, unet_sample_size//8,unet_sample_size//8).to(device=device, dtype=dtype),
			torch.randn(2, 1280, unet_sample_size//8,unet_sample_size//8).to(device=device, dtype=dtype),
			torch.randn(2, 1280, unet_sample_size//8,unet_sample_size//8).to(device=device, dtype=dtype),
			False,
		),
		output_path=cnet_path,
		ordered_input_names=["sample", 
							 "timestep", 
							 "encoder_hidden_states", 
							 "down_block_0",
							 "down_block_1",
							 "down_block_2",
							 "down_block_3",
							 "down_block_4",
							 "down_block_5",
							 "down_block_6",
							 "down_block_7",
							 "down_block_8",
							 "down_block_9",
							 "down_block_10",
							 "down_block_11",
							 "mid_block_additional_residual",
							 "return_dict"
							 ],
	
		output_names=["out_sample"],  # has to be different from "sample" for correct tracing
		dynamic_axes={
			"sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
			"timestep": {0: "batch"},
			"encoder_hidden_states": {0: "batch", 1: "sequence"},
			"down_block_0": {0: "batch", 2: "height", 3: "width"},
			"down_block_1": {0: "batch", 2: "height", 3: "width"},
			"down_block_2": {0: "batch", 2: "height", 3: "width"},
			"down_block_3": {0: "batch", 2: "height2", 3: "width2"},
			"down_block_4": {0: "batch", 2: "height2", 3: "width2"},
			"down_block_5": {0: "batch", 2: "height2", 3: "width2"},
			"down_block_6": {0: "batch", 2: "height4", 3: "width4"},
			"down_block_7": {0: "batch", 2: "height4", 3: "width4"},
			"down_block_8": {0: "batch", 2: "height4", 3: "width4"},
			"down_block_9": {0: "batch", 2: "height8", 3: "width8"},
			"down_block_10": {0: "batch", 2: "height8", 3: "width8"},
			"down_block_11": {0: "batch", 2: "height8", 3: "width8"},
			"mid_block_additional_residual": {0: "batch", 2: "height8", 3: "width8"},
		},
		opset=opset,
	)
	cnet_model_path = str(cnet_path.absolute().as_posix())
	cnet_dir = os.path.dirname(cnet_model_path)
	cnet = onnx.load(cnet_model_path)
	
	# clean up existing tensor files
	shutil.rmtree(cnet_dir)
	os.mkdir(cnet_dir)
	
	# collate external tensor files into one
	onnx.save_model(
		cnet,
		cnet_model_path,
		save_as_external_data=True,
		all_tensors_to_one_file=True,
		location="weights.pb",
		convert_attribute=False,
	)
	del pipe_cnet
	
	convert_to_fp16(cnet_model_path)

	# VAE ENCODER
	olive_export(model_path, output_path, "vae_encoder")

	# VAE DECODER
	olive_export(model_path, output_path, "vae_decoder")

	# SAFETY CHECKER
	safety_checker = None
	if pipeline.safety_checker is not None:
		olive_export(model_path, output_path, "safety_checker")
		pipeline.feature_extractor.save_pretrained(output_path / "feature_extractor")
		safety_checker = OnnxRuntimeModel.from_pretrained(output_path / "safety_checker")

	onnx_pipeline = OnnxStableDiffusionPipeline(
		vae_encoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_encoder"),
		vae_decoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_decoder"),
		text_encoder=OnnxRuntimeModel.from_pretrained(output_path / "text_encoder"),
		tokenizer=pipeline.tokenizer,
		unet=OnnxRuntimeModel.from_pretrained(output_path / "unet"),
		scheduler=pipeline.scheduler,
		safety_checker=safety_checker,
		feature_extractor=pipeline.feature_extractor,
		requires_safety_checker=safety_checker is not None,
	)

	onnx_pipeline.save_pretrained(output_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	if os.path.exists("./cache"):
		shutil.rmtree("./cache")

	if os.path.exists("./footprints"):
		shutil.rmtree("./footprints")

	parser.add_argument(
		"--model_path",
		type=str,
		required=True,
		help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
	)

	parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")
	parser.add_argument(
		"--opset",
		default=14,
		type=int,
		help="The version of the ONNX operator set to use.",
	)
	parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")

	args = parser.parse_args()
	print(args.output_path)
	convert_models(args.model_path, args.output_path, args.opset, args.fp16)
	print("SD: Done: ONNX")