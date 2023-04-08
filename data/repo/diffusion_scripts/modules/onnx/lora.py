from os import path
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import torch, safetensors

from onnx import ModelProto, load, numpy_helper
from onnx.external_data_helper import (
    set_external_data,
)
from onnxruntime import OrtValue

def buffer_external_data_tensors(
    model: ModelProto,
) -> Tuple[ModelProto, List[Tuple[str, OrtValue]]]:
    external_data = []
    for tensor in model.graph.initializer:
        name = tensor.name

        if tensor.HasField("raw_data"):
            npt = numpy_helper.to_array(tensor)
            orv = OrtValue.ortvalue_from_numpy(npt)
            external_data.append((name, orv))
            # mimic set_external_data
            set_external_data(tensor, location="foo.bin")
            tensor.name = name
            tensor.ClearField("raw_data")

    print("externalizing tensor: done!")

    return (model, external_data)


def fix_initializer_name(key: str):
    # lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_down.weight
    # lora, unet, up_block.3.attentions.2.transformer_blocks.0.attn2.to_out.0
    return key.replace(".", "_")


def fix_node_name(key: str):
    fixed_name = fix_initializer_name(key.replace("/", "_"))
    if fixed_name[0] == "_":
        return fixed_name[1:]
    else:
        return fixed_name


def blend_loras(
    base_name: Union[str, ModelProto],
    loras: str,
    model_type: Literal["text_encoder", "unet"],
    lora_weight: float
):
    # always load to CPU for blending
    dtype = torch.float32

    base_model = base_name if isinstance(base_name, ModelProto) else load(base_name)
    lora_model = safetensors.torch.load_file(loras, device="cpu")

    if model_type == "text_encoder":
        lora_prefix = "lora_te_"
    else:
        lora_prefix = f"lora_{model_type}_"

    blended: Dict[str, np.ndarray] = {}
    lora_name = loras

    print("blending LoRA from %s with weight of %s", lora_name, lora_weight)

    for key in lora_model.keys():
        if ".hada_w1_a" in key and lora_prefix in key:
            # LoHA
            base_key = key[: key.index(".hada_w1_a")].replace(lora_prefix, "")

            w1b_key = key.replace("hada_w1_a", "hada_w1_b")
            w2a_key = key.replace("hada_w1_a", "hada_w2_a")
            w2b_key = key.replace("hada_w1_a", "hada_w2_b")
            alpha_key = key[: key.index("hada_w1_a")] + "alpha"

            w1a_weight = lora_model[key].to(dtype=dtype)
            w1b_weight = lora_model[w1b_key].to(dtype=dtype)
            w2a_weight = lora_model[w2a_key].to(dtype=dtype)
            w2b_weight = lora_model[w2b_key].to(dtype=dtype)

            dim = w1a_weight.size()[0]
            alpha = lora_model.get(alpha_key, dim).to(dtype).numpy()

            try:
                weights = (w1a_weight @ w1b_weight) * (w2a_weight @ w2b_weight)
                np_weights = weights.numpy() * (alpha / dim)

                np_weights *= lora_weight
                if base_key in blended:
                    blended[base_key] += np_weights
                else:
                    blended[base_key] = np_weights

            except Exception as e:
                print("error blending weights for LoHA key %s: %s", base_key, e)

        elif ".lora_down" in key and lora_prefix in key:
            base_key = key[: key.index(".lora_down")].replace(lora_prefix, "")

            up_key = key.replace("lora_down", "lora_up")
            alpha_key = key[: key.index("lora_down")] + "alpha"

            down_weight = lora_model[key].to(dtype=dtype)
            up_weight = lora_model[up_key].to(dtype=dtype)

            dim = down_weight.size()[0]
            alpha = None

            alpha_key_value = lora_model.get(alpha_key, dim)

            if isinstance(alpha_key_value, int):
                alpha = alpha_key_value
            else:
                alpha = alpha_key_value.to(dtype).numpy()

            try:
                if len(down_weight.size()) == 2:
                    weights = up_weight @ down_weight
                    np_weights = weights.numpy() * (alpha / dim)
                elif len(down_weight.size()) == 4 and down_weight.shape[-2:] == (
                        1,
                        1,
                    ):
                    weights = (
                        (
                            up_weight.squeeze(3).squeeze(2)
                            @ down_weight.squeeze(3).squeeze(2)
                        )
                        .unsqueeze(2)
                        .unsqueeze(3)
                    )
                    np_weights = weights.numpy() * (alpha / dim)
                elif len(down_weight.size()) == 4 and down_weight.shape[-2:] == (
                        3,
                        3,
                    ):
                    # blend for nn.Conv2d 3x3
                    weights = torch.nn.functional.conv2d(
                        down_weight.permute(1, 0, 2, 3), up_weight
                    ).permute(1, 0, 2, 3)
                    np_weights = weights.numpy() * (alpha / dim)
                else:
                    print(
                        "unknown LoRA node type at %s: %s",
                        base_key,
                        up_weight.shape[-2:],
                    )
                    continue

                np_weights *= lora_weight
                if base_key in blended:
                    blended[base_key] += np_weights
                else:
                    blended[base_key] = np_weights

            except Exception as e:
                print("error blending weights for LoRA key %s: %s", base_key, e)

    print(
        "updating %s of %s initializers: %s",
        len(blended.keys()),
        len(base_model.graph.initializer),
        list(blended.keys()),
    )

    fixed_initializer_names = [
        fix_initializer_name(node.name) for node in base_model.graph.initializer
    ]
    print("fixed initializer names: %s", fixed_initializer_names)

    fixed_node_names = [fix_node_name(node.name) for node in base_model.graph.node]
    print("fixed node names: %s", fixed_node_names)

    for base_key, weights in blended.items():
        conv_key = base_key + "_Conv"
        matmul_key = base_key + "_MatMul"

        if conv_key in fixed_node_names:
            conv_idx = fixed_node_names.index(conv_key)
            conv_node = base_model.graph.node[conv_idx]

            # find weight initializer
            weight_name = [n for n in conv_node.input if ".weight" in n][0]
            weight_name = fix_initializer_name(weight_name)

            weight_idx = fixed_initializer_names.index(weight_name)
            weight_node = base_model.graph.initializer[weight_idx]

            # blending
            base_weights = numpy_helper.to_array(weight_node)

            if base_weights.shape[-2:] == (1, 1):
                if weights.shape[-2:] == (1, 1):
                    blended = base_weights.squeeze((3, 2)) + weights.squeeze((3, 2))
                else:
                    blended = base_weights.squeeze((3, 2)) + weights

                blended = np.expand_dims(blended, (2, 3))
            else:
                if base_weights.shape != weights.shape:
                    blended = base_weights + weights.reshape(base_weights.shape)
                else:
                    blended = base_weights + weights

            # replace the original initializer
            updated_node = numpy_helper.from_array(
                blended.astype(base_weights.dtype), weight_node.name
            )
            del base_model.graph.initializer[weight_idx]
            base_model.graph.initializer.insert(weight_idx, updated_node)
        elif matmul_key in fixed_node_names:
            weight_idx = fixed_node_names.index(matmul_key)
            weight_node = base_model.graph.node[weight_idx]

            # find the MatMul initializer
            matmul_name = [n for n in weight_node.input if "MatMul" in n][0]

            matmul_idx = fixed_initializer_names.index(matmul_name)
            matmul_node = base_model.graph.initializer[matmul_idx]

            # blending
            base_weights = numpy_helper.to_array(matmul_node)

            blended = base_weights + weights.transpose()

            # replace the original initializer
            updated_node = numpy_helper.from_array(
                blended.astype(base_weights.dtype), matmul_node.name
            )
            del base_model.graph.initializer[matmul_idx]
            base_model.graph.initializer.insert(matmul_idx, updated_node)

    return base_model
