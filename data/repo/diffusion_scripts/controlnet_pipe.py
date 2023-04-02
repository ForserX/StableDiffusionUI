from diffusers import UniPCMultistepScheduler, AutoencoderKL, OnnxRuntimeModel

import torch, os, time, numpy

from controlnet_aux import (
    OpenposeDetector,
    HEDdetector,
    MLSDdetector
)

import cv2, transformers

from huggingface_hub import snapshot_download

from modules.controlnet.laion_face_common import generate_annotation
from modules.controlnet.palette import ade_palette

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

def generateSegFromImage():
    image_processor = transformers.AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = transformers.UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
    
    in_img = Image.open(opt.img).convert('RGB')
    
    pixel_values = image_processor(in_img, return_tensors="pt").pixel_values
    
    with torch.no_grad():
      outputs = image_segmentor(pixel_values)
    
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[in_img.size[::-1]])[0]
    
    color_seg = numpy.zeros((seg.shape[0], seg.shape[1], 3), dtype=numpy.uint8) # height, width, 3
    
    palette = numpy.array(ade_palette())
    
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    
    color_seg = color_seg.astype(numpy.uint8)
    
    image = Image.fromarray(color_seg)

    image.save(opt.outfile)
    print(f"CN: Pose - {opt.outfile}")

def generateMLSDFromImage():

    mlsd = MLSDdetector.from_pretrained(opt.mdlpath)
    in_img = Image.open(opt.img)
    
    image = mlsd(in_img)

    image.save(opt.outfile)
    print(f"CN: Pose - {opt.outfile}")

def generateNormalFromImage():
    depth_estimator = transformers.pipeline("depth-estimation", model = snapshot_download("Intel/dpt-hybrid-midas", allow_patterns=["*.bin", "*.json"], cache_dir=opt.workdir))
    
    in_img = Image.open(opt.img)
    image = depth_estimator(in_img)['predicted_depth'][0]
    
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
    hed = HEDdetector.from_pretrained(opt.mdlpath)
    image = hed(in_img)

    image.save(opt.outfile)
    print(f"CN: Pose - {opt.outfile}")
    
def generateScribbleFromImage():
    print("processing generateScribbleFromImage()")
    in_img = Image.open(opt.img)
    hed = HEDdetector.from_pretrained(opt.mdlpath)
    image = hed(in_img, scribble=True)

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

def generateCannyFromImage():
    print("processing generateCanyFromImage()")
    in_img = Image.open(opt.img)
    image = numpy.array(in_img)

    low_threshold = 100
    high_threshold = 200
    
    image = cv2.Canny(image, low_threshold, high_threshold)
    Image.fromarray(image).save(opt.outfile)

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

if opt.mode == "MfI":
    generateMLSDFromImage()
    
if opt.mode == "SgfI":
    generateSegFromImage()
    
if opt.mode == "HfI":
    generateHedFromImage()
    
if opt.mode == "SfI":
    generateScribbleFromImage()

if opt.mode == "NfI":
    generateNormalFromImage()

if opt.mode == "DfI":
    generateDepthFromImage()

if opt.mode == "FfI":
    generateFaceFromImage()
    
if opt.mode == "CfI":
    generateCannyFromImage()

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
