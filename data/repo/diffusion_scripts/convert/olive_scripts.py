# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch, os
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers.models.clip.modeling_clip import CLIPTextModel


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batchsize, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batchsize = batchsize
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batchsize, self.torch_dtype), label


def get_base_model_name(model_name):
    return model_name


def is_lora_model(model_name):
    # TODO: might be a better way to detect (e.g. presence of LORA weights file)
    return False


# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------


def text_encoder_inputs(batchsize, torch_dtype):
    return torch.zeros((batchsize, 77), dtype=torch_dtype)


def text_encoder_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")
    return model


def text_encoder_conversion_inputs(model):
    return text_encoder_inputs(1, torch.int32)


def text_encoder_data_loader(data_dir, batchsize):
    return RandomDataLoader(text_encoder_inputs, batchsize, torch.int32)


# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------


def unet_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 4, 64, 64), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 77, 768), dtype=torch_dtype),
        "return_dict": False,
    }


def unet_load(model_name):
    from diffusers.models.attention_processor import AttnProcessor
    base_model_id = get_base_model_name(model_name)
    model = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    model.set_attn_processor(AttnProcessor())
    return model


def unet_conversion_inputs(model):
    return tuple(unet_inputs(1, torch.float32).values())


def unet_data_loader(data_dir, batchsize):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------


def vae_encoder_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 3, 512, 512), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_encoder_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    model.encoder.mid_block.attentions[0]._use_2_0_attn = False

    model.forward = lambda sample, return_dict: model.encode(sample, return_dict)[0].sample()
    return model


def vae_encoder_conversion_inputs(model):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


def vae_encoder_data_loader(data_dir, batchsize):
    return RandomDataLoader(vae_encoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------


def vae_decoder_inputs(batchsize, torch_dtype):
    return {
        "latent_sample": torch.rand((batchsize, 4, 64, 64), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_decoder_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    model.decoder.mid_block.attentions[0]._use_2_0_attn = False

    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# SAFETY CHECKER
# -----------------------------------------------------------------------------


def safety_checker_inputs(batchsize, torch_dtype):
    return {
        "clip_input": torch.rand((batchsize, 3, 224, 224), dtype=torch_dtype),
        "images": torch.rand((batchsize, 512, 512, 3), dtype=torch_dtype),
    }


def safety_checker_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = StableDiffusionSafetyChecker.from_pretrained(base_model_id, subfolder="safety_checker")
    model.forward = model.forward_onnx
    return model


def safety_checker_conversion_inputs(model):
    return tuple(safety_checker_inputs(1, torch.float32).values())


def safety_checker_data_loader(data_dir, batchsize):
    return RandomDataLoader(safety_checker_inputs, batchsize, torch.float16)