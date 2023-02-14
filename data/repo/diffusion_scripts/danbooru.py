import os
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
model.half()
model.cuda()

pic = Image.open(opt.img).convert("RGB").resize((512, 512))
a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

with torch.no_grad(), torch.autocast("cuda"):
    x = torch.from_numpy(a).cuda()

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
#processed = model.tag(pic)
#print(processed)