import os
import argparse
import clip
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
from torchvision.datasets import CIFAR100







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
        

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the image",
     )

    args = parser.parse_args()
    original_image = args.model_path
    texts = []
    

    for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    if name not in descriptions:
        continue

    image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
  
    plt.subplot(2, 4, len(images) + 1)
    plt.imshow(image)
    plt.title(f"{filename}\n{descriptions[name]}")
    plt.xticks([])
    plt.yticks([])

    original_images.append(image)
    images.append(preprocess(image))
    texts.append(descriptions[name])

plt.tight_layout()

    image_input = torch.tensor((original_image)).cuda()
    text_tokens = clip.tokenize([desc for desc in texts]).cuda()
         
    clip.available_models()
    
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size
    
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()
    
    
    cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
    
    text_descriptions = [f"{label}" for label in cifar100.classes]
    text_tokens = clip.tokenize(text_descriptions).cuda()
         
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
    

    print([lable for lable in top_labels])





    #parser.add_argument(
    #    "--model_path",
    #    type=str,
    #    required=True,
    #    help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    #)

    #parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")

     #parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    "--opset",
    #    default=14,
    #    type=int,
    #    help="The version of the ONNX operator set to use.",
    #)
    #parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")
    



    print("DeepDanBooru: Done")