import torch, time, os, numpy
from PIL import PngImagePlugin, Image

from safetensors.torch import load_file

from modules.onnx.custom_pipelines.pipeline_onnx_stable_diffusion_instruct_pix2pix import OnnxStableDiffusionInstructPix2PixPipeline
from modules.onnx.custom_pipelines.pipeline_onnx_stable_diffusion_controlnet import OnnxStableDiffusionControlNetPipeline

try:
	from onnxruntime import SessionOptions
	from modules.onnx.lora import blend_loras, buffer_external_data_tensors
	from modules.onnx.textual_inversion import blend_textual_inversions
	#import torch_directml

	ONNX_MODEL = "model.onnx"
except :
	pass

from diffusers import ( 
	OnnxStableDiffusionPipeline, 
	OnnxStableDiffusionImg2ImgPipeline,
	OnnxStableDiffusionInpaintPipeline, 
	OnnxRuntimeModel,
	StableDiffusionPipeline,
	StableDiffusionImg2ImgPipeline, 
	StableDiffusionInpaintPipeline,
	StableDiffusionControlNetPipeline, 
	StableDiffusionInstructPix2PixPipeline,
	ControlNetModel
)

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import (
	EulerAncestralDiscreteScheduler, 
	PNDMScheduler, 
	LMSDiscreteScheduler, 
	DDIMScheduler, 
	DPMSolverMultistepScheduler, 
	EulerDiscreteScheduler, 
	DDPMScheduler, 
	KDPM2DiscreteScheduler, 
	HeunDiscreteScheduler,
	UniPCMultistepScheduler
)

class Device:
	prov = "DmlExecutionProvider"
	device = "cpu"
	fptype = None
	device_str = "cpu"

	def __init__(self, device: str, fp):
		if device == "onnx":
			self.device = "onnx" #torch_directml.device(torch_directml.default_device())
		else:
			self.device = device
			
		self.device_str = device
		print(f"Current device: {self.device}")

		self.fptype = fp

	def LPW_Path(self):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		if self.device_str == "onnx":
			dir_path = dir_path + "/modules/onnx/lpw_stable_diffusion_onnx.py"
		else:
			dir_path = dir_path + "/modules/diffusers/lpw_stable_diffusion.py"
	
		return dir_path
	
	def GetPipe(self, Model: str, Mode: str, NSFW: bool):
		pipe = None
		nsfw_pipe = None
		
		if self.device_str == "onnx":
			if Mode == "pix2pix":
				if NSFW:
					pipe = OnnxStableDiffusionInstructPix2PixPipeline.from_pretrained("ForserX/instruct-pix2pix-onnx", provider=self.prov)
				else:
					pipe = OnnxStableDiffusionInstructPix2PixPipeline.from_pretrained("ForserX/instruct-pix2pix-onnx", provider=self.prov, safety_checker=None)
					
				pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
			else:
				if NSFW:
					safety_model = Model + "/safety_checker/"
					nsfw_pipe = OnnxRuntimeModel.from_pretrained(safety_model, provider=self.prov)
				print (Mode)    
				if Mode == "txt2img":
					pipe = OnnxStableDiffusionPipeline.from_pretrained(Model, custom_pipeline=self.LPW_Path(), provider=self.prov, safety_checker=nsfw_pipe)
				if Mode == "img2img":
					pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(Model, custom_pipeline=self.LPW_Path(), provider=self.prov, safety_checker=nsfw_pipe)
				if Mode == "inpaint":
					pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(Model, custom_pipeline=self.LPW_Path(), provider=self.prov, safety_checker=nsfw_pipe)
		else:
			if Mode == "pix2pix":
				if NSFW:
					pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=self.fptype)
				else:
					pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=self.fptype, safety_checker=None)
	
			else:
				if NSFW:
					safety_model = Model + "/safety_checker/"
					nsfw_pipe = StableDiffusionSafetyChecker.from_pretrained( safety_model, torch_dtype=self.fptype)
				print (Mode)      
				if Mode == "txt2img":
					pipe = StableDiffusionPipeline.from_pretrained(Model, custom_pipeline=self.LPW_Path(), torch_dtype=self.fptype, safety_checker=nsfw_pipe)
				if Mode == "img2img":
					pipe = StableDiffusionImg2ImgPipeline.from_pretrained(Model, custom_pipeline=self.LPW_Path(), torch_dtype=self.fptype, safety_checker=nsfw_pipe)
				if Mode == "inpaint":
					pipe = StableDiffusionInpaintPipeline.from_pretrained(Model, custom_pipeline=self.LPW_Path(), torch_dtype=self.fptype, safety_checker=nsfw_pipe)
					
		return pipe
	
	def GetPipeCN(self, Model: str, CNModel: str, NSFW: bool):
		pipe = None
		nsfw_pipe = None
		
		if self.device_str == "onnx":
			if NSFW:
				safety_model = Model + "/safety_checker/"
				nsfw_pipe = OnnxRuntimeModel.from_pretrained(safety_model, provider=self.prov)
	
			controlnet = OnnxRuntimeModel.from_pretrained(CNModel, provider="DmlExecutionProvider")
			
			cnet = OnnxRuntimeModel.from_pretrained(Model + "/cnet/", provider=self.prov)
			pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
				Model,
				unet=cnet,
				controlnet=controlnet,
				provider="DmlExecutionProvider",
				safety_checker=nsfw_pipe
			)
			
		else:
			print(CNModel)
			controlnet = ControlNetModel.from_pretrained(
				CNModel, torch_dtype=self.fptype
			)
	
			if NSFW:
				safety_model = Model + "/safety_checker/"
				nsfw_pipe = StableDiffusionSafetyChecker.from_pretrained(safety_model, torch_dtype=self.fptype)
			
			pipe = StableDiffusionControlNetPipeline.from_pretrained(
				Model,
				controlnet=controlnet,
				torch_dtype=self.fptype, 
				safety_checker=nsfw_pipe
			)
	
		return pipe
	
	def ApplyLoraONNX(self, opt, p_te_model, lora, alpha, pipe):
		blended_unet = blend_loras(opt.mdlpath + "/unet/" + ONNX_MODEL, lora, "unet", alpha)
		pipe.unet = self.ONNXProto2Runtime(blended_unet)

		blended_te = blend_loras(p_te_model, lora, "text_encoder", alpha)
		
		return blended_te

	def ApplyTE(self, p_model, te_name, alpha, pipe):
		pipe.tokenizer, prompt_tokens = blend_textual_inversions(p_model, pipe.tokenizer, te_name, alpha)

		return (p_model, prompt_tokens)
	
	def ONNXProto2Runtime(self, model, provider="DmlExecutionProvider"):
		(unet_model, unet_data) = buffer_external_data_tensors(model)
		unet_names, unet_values = zip(*unet_data)
		sess = SessionOptions()
		sess.add_external_initializers(list(unet_names), list(unet_values))
		
		out_model = OnnxRuntimeModel(OnnxRuntimeModel.load_model(unet_model.SerializeToString(), provider=self.prov, sess_options=sess))
		return out_model
	
	def ApplyLoRA(self, unet, te, LoraPath : str, strength: float):
		model_path = LoraPath
		state_dict = load_file(model_path, self.device)
	
		print ("lora strength = ", strength)
	
		LORA_PREFIX_UNET = 'lora_unet'
		LORA_PREFIX_TEXT_ENCODER = 'lora_te'
			
		visited = []
		
		# directly update weight in diffusers model
		for key in state_dict:
			
			# it is suggested to print out the key, it usually will be something like below
			# "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
			
			# as we have set the alpha beforehand, so just skip
			if '.alpha' in key or key in visited:
				continue
			
			LORA_PREFIX = None
	
			if 'text' in key:
				LORA_PREFIX = LORA_PREFIX_TEXT_ENCODER
				layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
				curr_layer = te
			else:
				LORA_PREFIX = LORA_PREFIX_UNET
				layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
				curr_layer = unet
		
			# find the target layer
			temp_name = layer_infos.pop(0)
			while len(layer_infos) > -1:
				try:
					curr_layer = curr_layer.__getattr__(temp_name)
					if len(layer_infos) > 0:
						temp_name = layer_infos.pop(0)
					elif len(layer_infos) == 0:
						break
				except Exception:
					if len(temp_name) > 0:
						temp_name += '_'+layer_infos.pop(0)
					else:
						temp_name = layer_infos.pop(0)
			
			# org_forward(x) + lora_up(lora_down(x)) * multiplier
			pair_keys = []
	
			if ".hada_w1_a" in key and LORA_PREFIX in key:
				# LoHA
						   
				t1_key = key.replace("hada_w1_a", "hada_t1")
				t2_key = key.replace("hada_w1_a", "hada_t2")
	
				w1b_key = key.replace("hada_w1_a", "hada_w1_b")
				w2a_key = key.replace("hada_w1_a", "hada_w2_a")
				w2b_key = key.replace("hada_w1_a", "hada_w2_b")
				alpha_key = key[: key.index("hada_w1_a")] + "alpha"
	
				w1a_weight = state_dict[key].to(self.device,self.fptype)
				w1b_weight = state_dict[w1b_key].to(self.device, self.fptype)
				w2a_weight = state_dict[w2a_key].to(self.device, self.fptype)
				w2b_weight = state_dict[w2b_key].to(self.device, self.fptype)
				
				t1_weight = state_dict.get(t1_key, None)
				t2_weight = state_dict.get(t2_key, None)
	
				dim = w1b_weight.shape[0]
				alpha = state_dict.get(alpha_key, dim).to(self.device, self.fptype)
				
				if t1_weight is not None and t2_weight is not None:
					t1_weight = t1_weight.to(self.device, self.fptype)
					t2_weight = t2_weight.to(self.device, self.fptype)
					
					weights_1 = torch.einsum(
						"i j k l, j r, i p -> p r k l",
						t1_weight,
						w1b_weight,
						w1a_weight,
					)
					weights_2 = torch.einsum(
						"i j k l, j r, i p -> p r k l",
						t2_weight,
						w2b_weight,
						w2a_weight,
					)
					weights = weights_1 * weights_2
					np_weights = weights * (alpha / dim)
				else:
					weights = (w1a_weight @ w1b_weight) * (w2a_weight @ w2b_weight)
					np_weights = weights * (alpha / dim)
	
				np_weights *= strength
			  
				
	
				curr_layer.weight.data += np_weights.reshape(curr_layer.weight.data.shape)
			else:            
	
				if 'lora_down' in key:
					pair_keys.append(key.replace('lora_down', 'lora_up'))
					pair_keys.append(key)
				elif 'lora_up' in key:
					pair_keys.append(key)
					pair_keys.append(key.replace('lora_up', 'lora_down'))
				
				# update weight
				if len(pair_keys) == 0:
					continue
				
				if len(state_dict[pair_keys[1]].shape) == 4:
				   
					if weight_down.shape[-2:] == (1, 1):
						
						weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(self.device, self.fptype)
						weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(self.device, self.fptype)
						alpha_key = key[: key.index("lora_down")] + "alpha"
						dim = weight_down.size()[0]
						alpha = state_dict.get(alpha_key, dim)                   
						k_weight = strength * (alpha / dim) 
						curr_layer.weight.data += k_weight * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
					else:
						
						weight_up = state_dict[pair_keys[0]].to(self.device, self.fptype)                   
						weight_down = state_dict[pair_keys[1]].permute(1, 0, 2, 3).to(self.device, self.fptype)
						alpha_key = key[: key.index("lora_down")] + "alpha"   
						dim = weight_down.size()[0]
						alpha = state_dict.get(alpha_key, dim)                    
						k_weight = strength * (alpha / dim)
						curr_layer.weight.data += k_weight * torch.nn.functional.conv2d(weight_down, weight_up).permute(1, 0, 2, 3)
				else:
					weight_up = state_dict[pair_keys[0]].to(self.device, self.fptype)
					weight_down = state_dict[pair_keys[1]].to(self.device, self.fptype)
					alpha_key = key[: key.index("lora_down")] + "alpha"   
					dim = weight_down.size()[0]
					alpha = state_dict.get(alpha_key, dim)                    
					k_weight = strength * (alpha / dim)
					curr_layer.weight.data += k_weight * torch.mm(weight_up, weight_down)
				
			 # update visited list
			for item in pair_keys:
				visited.append(item)
	
	def GetSampler(self, Pipe, SamplerName: str, ETA):
		eta = 0
		if SamplerName == "EulerAncestralDiscrete":
			Pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(Pipe.scheduler.config)
			eta = ETA
		if SamplerName == "EulerDiscrete":
			Pipe.scheduler = EulerDiscreteScheduler.from_config(Pipe.scheduler.config)
		if SamplerName == "PNDM":
			Pipe.scheduler = PNDMScheduler.from_config(Pipe.scheduler.config)
		if SamplerName == "DDIM":
			Pipe.scheduler = DDIMScheduler.from_config(Pipe.scheduler.config)
		if SamplerName == "DPMSolverMultistep":
			Pipe.scheduler = DPMSolverMultistepScheduler.from_config(Pipe.scheduler.config)
		if SamplerName == "LMSDiscrete":
			Pipe.scheduler = LMSDiscreteScheduler.from_config(Pipe.scheduler.config)
		if SamplerName == "DDPM":
			Pipe.scheduler = DDPMScheduler.from_config(Pipe.scheduler.config)
		if SamplerName == "DPMDiscrete":
			Pipe.scheduler = KDPM2DiscreteScheduler.from_config(Pipe.scheduler.config)
		if SamplerName == "HeunDiscrete":
			Pipe.scheduler = HeunDiscreteScheduler.from_config(Pipe.scheduler.config)
		if SamplerName == "UniPCMultistep":
			Pipe.scheduler = UniPCMultistepScheduler.from_config(Pipe.scheduler.config)
	
		return eta
	
	def MakeImage(self, pipe, mode : str, eta, prompt, prompt_neg, steps, width, height, seed, scale, image_guidance_scale, init_img_path = None, img_strength = 0.75, mask_img_path = None, outpath = "", batch_size = 1):
		start_time = time.time()
		
		seed = int(seed)
		print(f"Set seed to {seed}", flush=True)
		
		if not self.device_str == "onnx":
			rng = torch.Generator(self.device_str).manual_seed(seed)
		else: 
			if mode == "pix2pix":
				rng=numpy.random.seed(seed)
			else:
				rng = torch.Generator("cpu").manual_seed(seed)
	
		print(mode, flush=True)
		
		if mode == "txt2img":
			image=pipe(prompt=[prompt] * batch_size, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, generator=rng)
		
		if mode == "img2img":
			# Opt image
			img=Image.open(init_img_path).convert("RGB").resize((width, height))
			image=pipe(prompt=[prompt] * batch_size, image=img, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, strength=img_strength, generator=rng)
	
		if mode == "pix2pix":
			# Opt image
			img=Image.open(init_img_path).convert("RGB").resize((width, height))
			image=pipe(prompt=prompt, image=img, num_inference_steps=steps, guidance_scale=scale, image_guidance_scale=image_guidance_scale, negative_prompt=prompt_neg, eta=eta, generator=rng)
	
		if mode == "inpaint":
			img=Image.open(init_img_path).convert("RGB").resize((width, height))
			mask=Image.open(mask_img_path).convert("RGB").resize((width, height))
			image=pipe(prompt=[prompt] * batch_size, image=img, mask_image = mask, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, negative_prompt=prompt_neg, eta=eta, generator=rng)
		
		safe_end_time = time.time();
		# PNG MetaData
		info = PngImagePlugin.PngInfo()
		MetaText = "Prompt: {" + f"{prompt}" + "}, NegativePrompt: {" + f"{prompt_neg}" + "} " + f"\nSeed: {seed}, Steps: {steps}, Size: {width}x{height}, Mode: {mode}, CFG Scale: {scale}"
		info.add_text("XUI Metadata", MetaText)
		
		for i in range(len(image.images)):
			image.images[i].save(os.path.join(outpath, f"{time.time_ns()}_{i}.png"), 'PNG', pnginfo=info)
		
		print(f'Image generated in {(safe_end_time - start_time):.2f}s')
		image = None
	
	def ApplyArg(parser):
		parser.add_argument(
			"--model", type=str, help="Path to model checkpoint file", dest='mdlpath',
		)
		parser.add_argument(
			"--workdir", default=None, type=str, help="Path to model checkpoint file", dest='workdir',
		)
		parser.add_argument(
			"--width", type=int, help="Path to model checkpoint file", dest='width',
		)
		parser.add_argument(
			"--guidance_scale", type=float, help="Path to model checkpoint file", dest='guidance_scale',
		)
		parser.add_argument(
			"--height", type=int, help="Path to model checkpoint file", dest='height',
		)
		parser.add_argument(
			"--totalcount",
			type=int, help="Path to model checkpoint file", dest='totalcount',
		)
		parser.add_argument(
			"--steps",
			type=int, help="Path to model checkpoint file", dest='steps',
		)
		parser.add_argument(
			"--seed", type=int, help="Path to model checkpoint file", dest='seed',
		)
		parser.add_argument(
			"--imgscale",
			type=float, default=0.44, help="Path to model checkpoint file", dest='imgscale',
		)
		parser.add_argument(
			"--prompt_neg", type=str, help="Path to model checkpoint file", dest='prompt_neg',
		)
		parser.add_argument(
			"--prompt", type=str, help="Path to model checkpoint file", dest='prompt',
		)
		parser.add_argument(
			"--outpath",
			type=str, help="Output path", dest='outpath',
		)
		parser.add_argument(
			"--precision", type=str, help="precision type (fp16/fp32)", dest='precision',
		)
		parser.add_argument(
			"--mode", type=str, help="Specify generation mode", dest='mode',
		)
		parser.add_argument(
			"--img", type=str, default=None, help="Specify generation mode", dest='img',
		)
		parser.add_argument(
			"--imgmask", type=str, default=None, help="Specify generation image mask", dest='imgmask',
		)
		parser.add_argument(
			"--device", type=str, default="cuda", help="Specify generation mode device", dest='device',
		)
		parser.add_argument(
			"--scmode", default="eulera", help="Specify generation scmode", dest='scmode',
		)
		parser.add_argument(
			"--vae", help="Specify generation vae path", dest='vae',
		)
		parser.add_argument(
			"--eta", help="Eta", dest='eta', default=1.0,
		)
		parser.add_argument(
			"--nsfw", help="nsfw checker", dest='nsfw', default=False,
		)
		parser.add_argument(
			"--lora", help="lora checker", dest='lora', default=False,
		)
		parser.add_argument(
			"--dlora", help="lora checker", dest='dlora', default=False,
		)
		parser.add_argument(
			"--lora_path", type=str, help="Path to model LoRA file", dest='lora_path',
		)
		parser.add_argument(
			"--lora_strength", type=float, help="lora strength (alpha)", dest='lora_strength',
		)
		parser.add_argument(
			"--inversion", help="inversion path", dest='inversion', default=None,
		)
		parser.add_argument(
			"--hypernetwork", type=str, help="hypernetwork path", dest='hypernetwork', default=None,
		)
		parser.add_argument(
			"--cn_model", type=str, help="Path to model checkpoint file", dest='cn_model',
		)
		parser.add_argument(
			"--outfile", type=str, default="", help="Specify generation mode", dest='outfile',
		)
		parser.add_argument(
			"--pose", type=str, default="", help="input pose image", dest='pose',
		)
		parser.add_argument(
			"--strenght", type=float, default=0.45, help="strenght", dest='strenght',
		)
		parser.add_argument(
			"--img_strength", type=float, default=0.75, help="img_strength", dest='img_strength',
		)
		parser.add_argument(
			"--image_guidance_scale", type=float, default=1.5, help="image_guidance_scale", dest='image_guidance_scale',
		)