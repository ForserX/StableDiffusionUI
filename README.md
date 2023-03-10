# StableDiffusionUI Shark AMD and Nvidia
### What is it
Simple re-implement SD UI for AMD and Nvidia GPU 
![image](https://user-images.githubusercontent.com/13867290/222996644-77cfab99-6a1a-45e1-823e-9a1365e515db.png)
![image](https://user-images.githubusercontent.com/13867290/216797870-3f05fd70-41b0-41e5-b9ea-e7f41f294b65.png)


* Shark (AMD: Vulkan + MRIL/IREE)
* Diffuser ONNX (AMD: DirectML/DirectX 12)
* Diffuser (CPU/CUDA)

# How to install
## Pre-install
* Install python 3.10 + pip + venv
* __(SHARK Only)__ Install IREE drivers (Any driver newer than 23.2.1)

## Setup
* Start .exe file and select backend type

## Stargazers over time
[![Stargazers over time](https://starchart.cc/ForserX/StableDiffusionUI.svg)](https://starchart.cc/ForserX/StableDiffusionUI)

# Features
* Custom VAE (ONNX/CUDA)
* Lora (CUDA) 
* ControlNet Pose (CUDA/ONNX)
* DeepDanbooru tokens extractor (ONNX/CUDA)
* Converter for Diffusers/ONNX 
* Vulkan upscalers (ONNX/CUDA)
* Text2Image/Image2Image/Inpaint (ONNX/CUDA)
* Face restoration (ONNX/CUDA)
* Pix2Pix (CUDA/ONNX)
* * And other 
