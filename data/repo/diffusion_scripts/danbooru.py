import re

import torch
import argparse
from PIL import Image
import numpy as np
import tqdm
import danbooru_model

re_special = re.compile(r'([\\()])')

parser = argparse.ArgumentParser()
parser.add_argument(
    "--img",
    type=str,
    required=True,
    help="input Image",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Model Path",
)

opt = parser.parse_args()

model = danbooru_model.DeepDanbooruModel()
print(opt.model)

model.load_state_dict(torch.load(opt.model))
model.eval()

if torch.cuda.is_available(): 
    device = "cuda"
    dtype = torch.float16
    ntype =np.float32
else:
    device = "cpu"
    dtype = torch.float32
    ntype = np.float16
    
model.to(torch.device(device), dtype)

pic = Image.open(opt.img).convert("RGB").resize((512, 512))
a = np.expand_dims(np.array(pic, dtype=ntype), 0) / 255

with torch.no_grad():

    if torch.cuda.is_available(): 
        x = torch.from_numpy(a).cuda()
    else:
        x = torch.from_numpy(a).cpu()

    # first run
    y = model(x)[0].detach().cpu().numpy()

    # measure performance
    for n in tqdm.tqdm(range(10)):
        model(x)

output = []
for i, p in enumerate(y):
    if p >= 0.5:
        output.append(model.tags[i])
        
print(output,"DeepDanBooru: Finished!")