# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import PIL
import re
from transformers import CLIPFeatureExtractor, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import deprecate, logging, PIL_INTERPOLATION
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


logger = logging.get_logger(__name__)

re_attention = re.compile(
	r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
	re.X,
)


def parse_prompt_attention(text):
	"""
	Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
	Accepted tokens are:
	  (abc) - increases attention to abc by a multiplier of 1.1
	  (abc:3.12) - increases attention to abc by a multiplier of 3.12
	  [abc] - decreases attention to abc by a multiplier of 1.1
	  \( - literal character '('
	  \[ - literal character '['
	  \) - literal character ')'
	  \] - literal character ']'
	  \\ - literal character '\'
	  anything else - just text
	>>> parse_prompt_attention('normal text')
	[['normal text', 1.0]]
	>>> parse_prompt_attention('an (important) word')
	[['an ', 1.0], ['important', 1.1], [' word', 1.0]]
	>>> parse_prompt_attention('(unbalanced')
	[['unbalanced', 1.1]]
	>>> parse_prompt_attention('\(literal\]')
	[['(literal]', 1.0]]
	>>> parse_prompt_attention('(unnecessary)(parens)')
	[['unnecessaryparens', 1.1]]
	>>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
	[['a ', 1.0],
	 ['house', 1.5730000000000004],
	 [' ', 1.1],
	 ['on', 1.0],
	 [' a ', 1.1],
	 ['hill', 0.55],
	 [', sun, ', 1.1],
	 ['sky', 1.4641000000000006],
	 ['.', 1.1]]
	"""

	res = []
	round_brackets = []
	square_brackets = []

	round_bracket_multiplier = 1.1
	square_bracket_multiplier = 1 / 1.1

	def multiply_range(start_position, multiplier):
		for p in range(start_position, len(res)):
			res[p][1] *= multiplier

	for m in re_attention.finditer(text):
		text = m.group(0)
		weight = m.group(1)

		if text.startswith("\\"):
			res.append([text[1:], 1.0])
		elif text == "(":
			round_brackets.append(len(res))
		elif text == "[":
			square_brackets.append(len(res))
		elif weight is not None and len(round_brackets) > 0:
			multiply_range(round_brackets.pop(), float(weight))
		elif text == ")" and len(round_brackets) > 0:
			multiply_range(round_brackets.pop(), round_bracket_multiplier)
		elif text == "]" and len(square_brackets) > 0:
			multiply_range(square_brackets.pop(), square_bracket_multiplier)
		else:
			res.append([text, 1.0])

	for pos in round_brackets:
		multiply_range(pos, round_bracket_multiplier)

	for pos in square_brackets:
		multiply_range(pos, square_bracket_multiplier)

	if len(res) == 0:
		res = [["", 1.0]]

	# merge runs of identical weights
	i = 0
	while i + 1 < len(res):
		if res[i][1] == res[i + 1][1]:
			res[i][0] += res[i + 1][0]
			res.pop(i + 1)
		else:
			i += 1

	return res


def get_prompts_with_weights(pipe, prompt: List[str], max_length: int):
	r"""
	Tokenize a list of prompts and return its tokens with weights of each token.

	No padding, starting or ending token is included.
	"""
	tokens = []
	weights = []
	truncated = False
	for text in prompt:
		texts_and_weights = parse_prompt_attention(text)
		text_token = []
		text_weight = []
		for word, weight in texts_and_weights:
			# tokenize and discard the starting and the ending token
			token = pipe.tokenizer(word, return_tensors="np").input_ids[0, 1:-1]
			text_token += list(token)
			# copy the weight by length of token
			text_weight += [weight] * len(token)
			# stop if the text is too long (longer than truncation limit)
			if len(text_token) > max_length:
				truncated = True
				break
		# truncate
		if len(text_token) > max_length:
			truncated = True
			text_token = text_token[:max_length]
			text_weight = text_weight[:max_length]
		tokens.append(text_token)
		weights.append(text_weight)
	if truncated:
		logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
	return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
	r"""
	Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
	"""
	max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
	weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
	for i in range(len(tokens)):
		tokens[i] = [bos] + tokens[i] + [pad] * (max_length - 1 - len(tokens[i]) - 1) + [eos]
		if no_boseos_middle:
			weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
		else:
			w = []
			if len(weights[i]) == 0:
				w = [1.0] * weights_length
			else:
				for j in range(max_embeddings_multiples):
					w.append(1.0)  # weight for starting token in this chunk
					w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
					w.append(1.0)  # weight for ending token in this chunk
				w += [1.0] * (weights_length - len(w))
			weights[i] = w[:]

	return tokens, weights

def get_unweighted_text_embeddings(
	pipe,
	text_input: np.array,
	chunk_length: int,
	no_boseos_middle: Optional[bool] = True,
):
	"""
	When the length of tokens is a multiple of the capacity of the text encoder,
	it should be split into chunks and sent to the text encoder individually.
	"""
	max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
	if max_embeddings_multiples > 1:
		text_embeddings = []
		for i in range(max_embeddings_multiples):
			# extract the i-th chunk
			text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].copy()

			# cover the head and the tail by the starting and the ending tokens
			text_input_chunk[:, 0] = text_input[0, 0]
			text_input_chunk[:, -1] = text_input[0, -1]

			text_embedding = pipe.text_encoder(input_ids=text_input_chunk)[0]

			if no_boseos_middle:
				if i == 0:
					# discard the ending token
					text_embedding = text_embedding[:, :-1]
				elif i == max_embeddings_multiples - 1:
					# discard the starting token
					text_embedding = text_embedding[:, 1:]
				else:
					# discard both starting and ending tokens
					text_embedding = text_embedding[:, 1:-1]

			text_embeddings.append(text_embedding)
		text_embeddings = np.concatenate(text_embeddings, axis=1)
	else:
		text_embeddings = pipe.text_encoder(input_ids=text_input)[0]
	return text_embeddings


def get_weighted_text_embeddings(
	pipe,
	prompt: Union[str, List[str]],
	uncond_prompt: Optional[Union[str, List[str]]] = None,
	max_embeddings_multiples: Optional[int] = 4,
	no_boseos_middle: Optional[bool] = False,
	skip_parsing: Optional[bool] = False,
	skip_weighting: Optional[bool] = False,
	**kwargs,
):
	r"""
	Prompts can be assigned with local weights using brackets. For example,
	prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
	and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

	Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

	Args:
		pipe (`OnnxStableDiffusionPipeline`):
			Pipe to provide access to the tokenizer and the text encoder.
		prompt (`str` or `List[str]`):
			The prompt or prompts to guide the image generation.
		uncond_prompt (`str` or `List[str]`):
			The unconditional prompt or prompts for guide the image generation. If unconditional prompt
			is provided, the embeddings of prompt and uncond_prompt are concatenated.
		max_embeddings_multiples (`int`, *optional*, defaults to `1`):
			The max multiple length of prompt embeddings compared to the max output length of text encoder.
		no_boseos_middle (`bool`, *optional*, defaults to `False`):
			If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
			ending token in each of the chunk in the middle.
		skip_parsing (`bool`, *optional*, defaults to `False`):
			Skip the parsing of brackets.
		skip_weighting (`bool`, *optional*, defaults to `False`):
			Skip the weighting. When the parsing is skipped, it is forced True.
	"""
	max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
	if isinstance(prompt, str):
		prompt = [prompt]

	if not skip_parsing:
		prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt, max_length - 2)
		if uncond_prompt is not None:
			if isinstance(uncond_prompt, str):
				uncond_prompt = [uncond_prompt]
			uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, uncond_prompt, max_length - 2)
	else:
		prompt_tokens = [
			token[1:-1]
			for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True, return_tensors="np").input_ids
		]
		prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
		if uncond_prompt is not None:
			if isinstance(uncond_prompt, str):
				uncond_prompt = [uncond_prompt]
			uncond_tokens = [
				token[1:-1]
				for token in pipe.tokenizer(
					uncond_prompt,
					max_length=max_length,
					truncation=True,
					return_tensors="np",
				).input_ids
			]
			uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

	# round up the longest length of tokens to a multiple of (model_max_length - 2)
	max_length = max([len(token) for token in prompt_tokens])
	if uncond_prompt is not None:
		max_length = max(max_length, max([len(token) for token in uncond_tokens]))

	max_embeddings_multiples = min(
		max_embeddings_multiples,
		(max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
	)
	max_embeddings_multiples = max(1, max_embeddings_multiples)
	max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

	# pad the length of tokens and weights
	bos = pipe.tokenizer.bos_token_id
	eos = pipe.tokenizer.eos_token_id
	pad = getattr(pipe.tokenizer, "pad_token_id", eos)
	prompt_tokens, prompt_weights = pad_tokens_and_weights(
		prompt_tokens,
		prompt_weights,
		max_length,
		bos,
		eos,
		pad,
		no_boseos_middle=no_boseos_middle,
		chunk_length=pipe.tokenizer.model_max_length,
	)
	prompt_tokens = np.array(prompt_tokens, dtype=np.int32)
	if uncond_prompt is not None:
		uncond_tokens, uncond_weights = pad_tokens_and_weights(
			uncond_tokens,
			uncond_weights,
			max_length,
			bos,
			eos,
			pad,
			no_boseos_middle=no_boseos_middle,
			chunk_length=pipe.tokenizer.model_max_length,
		)
		uncond_tokens = np.array(uncond_tokens, dtype=np.int32)

	# get the embeddings
	text_embeddings = get_unweighted_text_embeddings(
		pipe,
		prompt_tokens,
		pipe.tokenizer.model_max_length,
		no_boseos_middle=no_boseos_middle,
	)
	prompt_weights = np.array(prompt_weights, dtype=text_embeddings.dtype)
	if uncond_prompt is not None:
		uncond_embeddings = get_unweighted_text_embeddings(
			pipe,
			uncond_tokens,
			pipe.tokenizer.model_max_length,
			no_boseos_middle=no_boseos_middle,
		)
		uncond_weights = np.array(uncond_weights, dtype=uncond_embeddings.dtype)

	# assign weights to the prompts and normalize in the sense of mean
	# TODO: should we normalize by chunk or in a whole (current implementation)?
	if (not skip_parsing) and (not skip_weighting):
		previous_mean = text_embeddings.mean(axis=(-2, -1))
		text_embeddings *= prompt_weights[:, :, None]
		text_embeddings *= (previous_mean / text_embeddings.mean(axis=(-2, -1)))[:, None, None]
		if uncond_prompt is not None:
			previous_mean = uncond_embeddings.mean(axis=(-2, -1))
			uncond_embeddings *= uncond_weights[:, :, None]
			uncond_embeddings *= (previous_mean / uncond_embeddings.mean(axis=(-2, -1)))[:, None, None]

	# For classifier free guidance, we need to do two forward passes.
	# Here we concatenate the unconditional and text embeddings into a single batch
	# to avoid doing two forward passes
	if uncond_prompt is not None:
		return text_embeddings, uncond_embeddings

	return text_embeddings


class OnnxStableDiffusionControlNetPipeline(DiffusionPipeline):
	vae_encoder: OnnxRuntimeModel
	vae_decoder: OnnxRuntimeModel
	text_encoder: OnnxRuntimeModel
	tokenizer: CLIPTokenizer
	unet: OnnxRuntimeModel
	controlnet: OnnxRuntimeModel
	scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]
	safety_checker: OnnxRuntimeModel
	feature_extractor: CLIPFeatureExtractor

	_optional_components = ["safety_checker", "feature_extractor"]

	def __init__(
		self,
		vae_encoder: OnnxRuntimeModel,
		vae_decoder: OnnxRuntimeModel,
		text_encoder: OnnxRuntimeModel,
		tokenizer: CLIPTokenizer,
		unet: OnnxRuntimeModel,
		controlnet: OnnxRuntimeModel,
		scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
		safety_checker: OnnxRuntimeModel,
		feature_extractor: CLIPFeatureExtractor,
		requires_safety_checker: bool = True,
	):
		super().__init__()

		if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
			deprecation_message = (
				f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
				f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
				"to update the config accordingly as leaving `steps_offset` might led to incorrect results"
				" in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
				" it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
				" file"
			)
			deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
			new_config = dict(scheduler.config)
			new_config["steps_offset"] = 1
			scheduler._internal_dict = FrozenDict(new_config)

		if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
			deprecation_message = (
				f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
				" `clip_sample` should be set to False in the configuration file. Please make sure to update the"
				" config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
				" future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
				" nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
			)
			deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
			new_config = dict(scheduler.config)
			new_config["clip_sample"] = False
			scheduler._internal_dict = FrozenDict(new_config)

		if safety_checker is None and requires_safety_checker:
			logger.warning(
				f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
				" that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
				" results in services or applications open to the public. Both the diffusers team and Hugging Face"
				" strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
				" it only for use-cases that involve analyzing network behavior or auditing its results. For more"
				" information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
			)

		if safety_checker is not None and feature_extractor is None:
			raise ValueError(
				"Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
				" checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
			)

		self.register_modules(
			vae_encoder=vae_encoder,
			vae_decoder=vae_decoder,
			text_encoder=text_encoder,
			tokenizer=tokenizer,
			unet=unet,
			controlnet=controlnet,
			scheduler=scheduler,
			safety_checker=safety_checker,
			feature_extractor=feature_extractor,
		)
		self.register_to_config(requires_safety_checker=requires_safety_checker)
		
		
	def _default_height_width(self, height, width, image):
		if isinstance(image, list):
			image = image[0]

		if height is None:
			if isinstance(image, PIL.Image.Image):
				height = image.height
			elif isinstance(image, np.ndarray):
				height = image.shape[3]

			height = (height // 8) * 8  # round down to nearest multiple of 8

		if width is None:
			if isinstance(image, PIL.Image.Image):
				width = image.width
			elif isinstance(image, np.ndarray):
				width = image.shape[2]

			width = (width // 8) * 8  # round down to nearest multiple of 8

		return height, width
		
	def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, dtype):
		if not isinstance(image, np.ndarray):
			if isinstance(image, PIL.Image.Image):
				image = [image]

			if isinstance(image[0], PIL.Image.Image):
				images = []

				for image_ in image:
					image_ = image_.convert("RGB")
					image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
					image_ = np.array(image_)
					image_ = image_[None, :]
					images.append(image_)

				image = images

				image = np.concatenate(image, axis=0)
				image = np.array(image).astype(np.float32) / 255.0
				image = image.transpose(0, 3, 1, 2)
				image = torch.from_numpy(image)
			elif isinstance(image[0], np.ndarray):
				image = np.concatenate(image, axis=0)
				image = torch.from_numpy(image)

		image_batch_size = image.shape[0]

		if image_batch_size == 1:
			repeat_by = batch_size
		else:
			# image batch size is the same as prompt batch size
			repeat_by = num_images_per_prompt

		image = image.repeat_interleave(repeat_by, dim=0)

		return image
		
	# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
	def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
		shape = (batch_size, num_channels_latents, height // 8, width // 8)
		if isinstance(generator, list) and len(generator) != batch_size:
			raise ValueError(
				f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
				f" size of {batch_size}. Make sure the batch size matches the length of the generators."
			)

		if latents is None:
			latents = generator.randn(*shape).astype(dtype)


		# scale the initial noise by the standard deviation required by the scheduler
		latents = latents * self.scheduler.init_noise_sigma
		return latents
	# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
	def prepare_extra_step_kwargs(self, generator, eta, torch_gen):
		# prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
		# eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
		# eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
		# and should be between [0, 1]

		accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
		extra_step_kwargs = {}
		if accepts_eta:
			extra_step_kwargs["eta"] = eta

		# check if the scheduler accepts generator
		accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
		if accepts_generator:
			extra_step_kwargs["generator"] = torch_gen
		return extra_step_kwargs
	
	def _encode_prompt(
		self,
		prompt,
		num_images_per_prompt,
		do_classifier_free_guidance,
		negative_prompt,
		max_embeddings_multiples,
	):
		r"""
		Encodes the prompt into text encoder hidden states.

		Args:
			prompt (`str` or `list(int)`):
				prompt to be encoded
			num_images_per_prompt (`int`):
				number of images that should be generated per prompt
			do_classifier_free_guidance (`bool`):
				whether to use classifier free guidance or not
			negative_prompt (`str` or `List[str]`):
				The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
				if `guidance_scale` is less than `1`).
			max_embeddings_multiples (`int`, *optional*, defaults to `3`):
				The max multiple length of prompt embeddings compared to the max output length of text encoder.
		"""
		batch_size = len(prompt) if isinstance(prompt, list) else 1

		if negative_prompt is None:
			negative_prompt = [""] * batch_size
		elif isinstance(negative_prompt, str):
			negative_prompt = [negative_prompt] * batch_size
		if batch_size != len(negative_prompt):
			raise ValueError(
				f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
				f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
				" the batch size of `prompt`."
			)

		text_embeddings, uncond_embeddings = get_weighted_text_embeddings(
			pipe=self,
			prompt=prompt,
			uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
			max_embeddings_multiples=max_embeddings_multiples,
		)

		text_embeddings = text_embeddings.repeat(num_images_per_prompt, 0)
		if do_classifier_free_guidance:
			uncond_embeddings = uncond_embeddings.repeat(num_images_per_prompt, 0)
			text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

		return text_embeddings

	def __call__(
		self,
		prompt: Union[str, List[str]],
		image: Union[np.ndarray, PIL.Image.Image] = None,
		height: Optional[int] = None,
		width: Optional[int] = None,
		num_inference_steps: Optional[int] = 50,
		guidance_scale: Optional[float] = 7.5,
		negative_prompt: Optional[Union[str, List[str]]] = None,
		num_images_per_prompt: Optional[int] = 1,
		eta: Optional[float] = 0.0,
		generator: Optional[np.random.RandomState] = None,
		latents: Optional[np.ndarray] = None,
		output_type: Optional[str] = "pil",
		return_dict: bool = True,
		callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
		callback_steps: int = 1,
		controlnet_conditioning_scale: float = 1.0,
	):
		if isinstance(prompt, str):
			batch_size = 1
		elif isinstance(prompt, list):
			batch_size = len(prompt)
		else:
			raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
			
			
		if generator:
			torch_seed = generator.randint(2147483647)
			torch_gen = torch.Generator().manual_seed(torch_seed)
		else:
			generator = np.random
			torch_gen = None
			
		height, width = self._default_height_width(height, width, image)

		if height % 8 != 0 or width % 8 != 0:
			raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

		if (callback_steps is None) or (
			callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
		):
			raise ValueError(
				f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
				f" {type(callback_steps)}."
			)

		# here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
		# of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
		# corresponds to doing no classifier free guidance.
		do_classifier_free_guidance = guidance_scale > 1.0
		
		max_embeddings_multiples = 3
		prompt_embeds = self._encode_prompt(
			prompt, num_images_per_prompt, do_classifier_free_guidance,
            negative_prompt,
            max_embeddings_multiples,
		)
		
		# 4. Prepare image
		image = self.prepare_image(
			image,
			width,
			height,
			batch_size * num_images_per_prompt,
			num_images_per_prompt,
			np.float32,
		).numpy()
		
		if do_classifier_free_guidance:
			image = np.concatenate([image] * 2)

		# get the initial random noise unless the user supplied it
		latents_dtype = prompt_embeds.dtype
		
		num_channels_latents = 4
		latents = self.prepare_latents(
			batch_size * num_images_per_prompt,
			num_channels_latents,
			height,
			width,
			latents_dtype,
			generator,
			latents,
		)
		
		# set timesteps
		self.scheduler.set_timesteps(num_inference_steps)
		timesteps = self.scheduler.timesteps

		# prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
		# eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
		# eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
		# and should be between [0, 1]
		extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta, torch_gen)

		timestep_dtype = next(
			(input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
		)
		timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]
		
		num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

		with self.progress_bar(total=num_inference_steps) as progress_bar:
			for i, t in enumerate(timesteps):
				# expand the latents if we are doing classifier free guidance
				latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
				latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
				latent_model_input = latent_model_input.cpu().numpy()
				
				timestep = np.array([t], dtype=timestep_dtype)
				
				try:
					blocksamples = self.controlnet(
						sample=latent_model_input,
						timestep=timestep,
						encoder_hidden_states=prompt_embeds,
						controlnet_cond=image,
					    conditioning_scale=1.0
					)	
				except :
					blocksamples = self.controlnet(
						sample=latent_model_input,
						timestep=timestep,
						encoder_hidden_states=prompt_embeds,
						controlnet_cond=image
					)	
				
				mid_block_res_sample=blocksamples[12]
				down_block_res_samples=(
					blocksamples[0],
					blocksamples[1],
					blocksamples[2],
					blocksamples[3],
					blocksamples[4],
					blocksamples[5],
					blocksamples[6],
					blocksamples[7],
					blocksamples[8],
					blocksamples[9],
					blocksamples[10],
					blocksamples[11],
				)
				
				down_block_res_samples = [
					down_block_res_sample * controlnet_conditioning_scale
					for down_block_res_sample in down_block_res_samples
				]
				mid_block_res_sample *= controlnet_conditioning_scale

				# predict the noise residual
				
				noise_pred = self.unet(
					sample=latent_model_input,
					timestep=timestep,
					encoder_hidden_states=prompt_embeds,
					down_block_0=down_block_res_samples[0],
					down_block_1=down_block_res_samples[1],
					down_block_2=down_block_res_samples[2],
					down_block_3=down_block_res_samples[3],
					down_block_4=down_block_res_samples[4],
					down_block_5=down_block_res_samples[5],
					down_block_6=down_block_res_samples[6],
					down_block_7=down_block_res_samples[7],
					down_block_8=down_block_res_samples[8],
					down_block_9=down_block_res_samples[9],
					down_block_10=down_block_res_samples[10],
					down_block_11=down_block_res_samples[11],
					mid_block_additional_residual=mid_block_res_sample
				)
				noise_pred = noise_pred[0]

				# perform guidance
				if do_classifier_free_guidance:
					noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
					noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

				# compute the previous noisy sample x_t -> x_t-1
				scheduler_output = self.scheduler.step(
					torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
				)
				latents = scheduler_output.prev_sample.numpy()

				# call the callback, if provided
				if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
					progress_bar.update()
					if callback is not None and i % callback_steps == 0:
						callback(i, t, latents)

		latents = 1 / 0.18215 * latents
		# image = self.vae_decoder(latent_sample=latents)[0]
		# it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
		image = np.concatenate(
			[self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
		)

		image = np.clip(image / 2 + 0.5, 0, 1)
		image = image.transpose((0, 2, 3, 1))

		if self.safety_checker is not None:
			safety_checker_input = self.feature_extractor(
				self.numpy_to_pil(image), return_tensors="np"
			).pixel_values.astype(image.dtype)

			images, has_nsfw_concept = [], []
			for i in range(image.shape[0]):
				image_i, has_nsfw_concept_i = self.safety_checker(
					clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
				)
				images.append(image_i)
				has_nsfw_concept.append(has_nsfw_concept_i[0])
			image = np.concatenate(images)
		else:
			has_nsfw_concept = None

		if output_type == "pil":
			image = self.numpy_to_pil(image)

		if not return_dict:
			return (image, has_nsfw_concept)

		return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
