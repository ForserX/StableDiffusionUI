from controlnet_aux import OpenposeDetector
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", type=str, help="Path to model checkpoint file", dest='model',
)
parser.add_argument(
    "--img", type=str, default="", help="Specify generation mode", dest='img',
)

parser.add_argument(
    "--outfile", type=str, default="", help="Specify generation mode", dest='outfile',
)

opt = parser.parse_args()

model = OpenposeDetector.from_pretrained(opt.model)
in_img = Image.open(opt.img)

img = model(in_img)
img.save(opt.outfile)

print(f"CN: Pose - {opt.outfile}")