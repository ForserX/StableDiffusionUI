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

import argparse
import os
import shutil
from pathlib import Path

import torch
from torch.onnx import export

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from diffusers import ControlNetModel

from packaging import version
from diffusers.models.cross_attention import CrossAttnProcessor

is_torch_less_2_0 = version.parse(version.parse(torch.__version__).base_version) < version.parse("2.0")

def convert_to_fp16(
    model_path
):
    '''Converts an ONNX model on disk to FP16'''
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
    opset
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


@torch.no_grad()
def convert_models(model_path: str, output_path: str, opset: int, attention_slicing: str):
    controlnet = ControlNetModel.from_pretrained(model_path)
    output_path = Path(output_path)
    if attention_slicing is not None:
        print("Attention slicing: " + attention_slicing)
        controlnet.set_attention_slice(attention_slicing)
        
    # UNET
    if is_torch_less_2_0 == False:
        controlnet.set_attn_processor(CrossAttnProcessor())

    dtype=torch.float32
    device = "cpu"
        
    cnet_path = output_path / "cnet" / "model.onnx"
    onnx_export(
        controlnet,
        model_args=(
            torch.randn(2, 4, 64, 64).to(device=device, dtype=dtype),
            torch.randn(2).to(device=device, dtype=dtype),
            torch.randn(2, 77, 768).to(device=device, dtype=dtype),
            torch.randn(2, 3, 512,512).to(device=device, dtype=dtype),
        ),
        output_path=cnet_path,
        ordered_input_names=["sample", "timestep", "encoder_hidden_states", "controlnet_cond","return_dict"],
        output_names=["down_block_res_samples", "mid_block_res_sample"],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "controlnet_cond": {0: "batch", 2: "height", 3: "width"}
        },
        opset=opset,
    )
    
    cnet_path_model_path = str(cnet_path.absolute().as_posix())
    convert_to_fp16(cnet_path_model_path)
    
    
    print("ONNX controlnet saved to ", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--opset",
        default=15,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    
    parser.add_argument(
        "--attention-slicing",
        choices={"auto","max"},
        type=str,
        help="Attention slicing, off by default. Can be set to auto. Reduces amount of VRAM used."
    )

    args = parser.parse_args()

    convert_models(args.model_path, args.output_path, args.opset, args.attention_slicing)