from diffusers import UniPCMultistepScheduler, AutoencoderKL, OnnxRuntimeModel

import torch, os, time, numpy

from controlnet_aux import (
    OpenposeDetector,
    HEDdetector
)

import cv2, transformers

from modules.cn_face.laion_face_common import generate_annotation

from PIL import Image
import argparse

from sd_xbackend import (
    GetPipeCN,
    ApplyArg,
    ApplyLoRA
)

parser = argparse.ArgumentParser()
ApplyArg(parser)
opt = parser.parse_args()

prov = "DmlExecutionProvider"
pipe = None
image = None

if opt.precision == "fp16":
    fptype = torch.float16
else:
    fptype = torch.float32

def apply_mediapipe_face(image, max_faces: int = 1, min_confidence: float = 0.5):
    return generate_annotation(image, max_faces, min_confidence)

def generateFaceFromImage():
    img = cv2.imread(opt.img)
    out = apply_mediapipe_face(img)
    Image.fromarray(out).save(opt.outfile)

print(opt)

def generateNormalFromImage():
    depth_estimator = transformers.pipeline("depth-estimation", model = "Intel/dpt-hybrid-midas")

    image = depth_estimator(image)['predicted_depth'][0]
    
    image = image.numpy()
    
    image_depth = image.copy()
    image_depth -= numpy.min(image_depth)
    image_depth /= numpy.max(image_depth)
    
    bg_threhold = 0.4
    
    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0
    
    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0
    
    z = numpy.ones_like(x) * numpy.pi * 2.0
    
    image = numpy.stack([x, y, z], axis=2)
    image /= numpy.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
    image = (image * 127.5 + 127.5).clip(0, 255).astype(numpy.uint8)
    image = Image.fromarray(image)
    
    image.save(opt.outfile)
    print(f"CN: Pose - {opt.outfile}")

def generateHedFromImage():
    print("processing generateHedFromImage()")
    in_img = Image.open(opt.img)
    hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
    image = hed(in_img)

    image.save(opt.outfile)
    print(f"CN: Pose - {opt.outfile}")

def generateDepthFromImage():
    print("processing generateDepthFromImage()")
    in_img = Image.open(opt.img)
    depth_estimator = transformers.pipeline('depth-estimation')

    image = depth_estimator(in_img)['depth']
    image = numpy.array(image)
    image = image[:, :, None]
    image = numpy.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    
    image.save(opt.outfile)
    print(f"CN: Pose - {opt.outfile}")

def generateCanyFromImage():
    print("processing generateCanyFromImage()")
    in_img = Image.open(opt.img)
    
    low_threshold = 100
    high_threshold = 200
    
    image = cv2.Canny(in_img, low_threshold, high_threshold)
    image.save(opt.outfile)

    print(f"CN: Pose - {opt.outfile}")
    
def generatePoseFromImage():
    print("processing generatePoseFromImage()")
    model = OpenposeDetector.from_pretrained(opt.mdlpath)
    in_img = Image.open(opt.img)
    
    img = model(in_img)
    img.save(opt.outfile)
    print(f"CN: Pose - {opt.outfile}")

def generateImageFromPose ():
    print("processing generateImageFromPose()")

    if opt.mode == "IfPONNX":
        generator = numpy.random.seed(opt.seed)
    else:
        generator = torch.Generator(device=opt.device).manual_seed(opt.seed)
        
    output = pipe(
        opt.prompt,
        image,
        opt.height,
        opt.width, 
        negative_prompt = opt.prompt_neg,
        generator = generator,
        num_inference_steps=opt.steps
    ).images[0]
    
    output.save(os.path.join(opt.outfile, f"{time.time_ns()}.png"), 'PNG')
    print("CN: Image from Pose: done!")

if opt.mode == "DfI":
    generateDepthFromImage()

if opt.mode == "FfI":
    generateFaceFromImage()
    
if opt.mode == "CfI":
    generateCanyFromImage()

if opt.mode == "PfI":
    generatePoseFromImage()

elif opt.mode == "IfPONNX":
    print("CN: ONNX initial")
    image = Image.open(opt.pose)
    image.convert("RGB").resize((opt.width, opt.height))

    pipe = GetPipeCN(opt.mdlpath, opt.cn_model, opt.nsfw, opt.precision == "fp16", True)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if not opt.vae == "default":
        pipe.vae_decoder = OnnxRuntimeModel.from_pretrained(opt.vae + "/vae_decoder", provider=prov)
        
    print("CN: Model loaded")
    for i in range(opt.totalcount):
        generateImageFromPose()
        opt.seed = opt.seed + 1

elif opt.mode == "IfP":
    print("CN: initial")
    image = Image.open(opt.pose)
    image.convert("RGB").resize((opt.width, opt.height))

    pipe = GetPipeCN(opt.mdlpath, opt.cn_model, opt.nsfw, opt.precision == "fp16")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if not opt.vae == "default":
        pipe.vae = AutoencoderKL.from_pretrained(opt.vae + "/vae", torch_dtype=fptype)
        
    pipe.to(opt.device)

    #if opt.device == "cpu":
    #    pipe.enable_model_cpu_offload()

    # LoRA magic
    if opt.lora:
        ApplyLoRA(pipe, opt.lora_path, opt.device, opt.precision == "fp16", 0.75)

    print("CN: Model loaded")
    for i in range(opt.totalcount):
        generateImageFromPose()
        opt.seed = opt.seed + 1
