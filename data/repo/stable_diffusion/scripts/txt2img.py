import os

os.environ["AMD_ENABLE_LLPC"] = "1"

import json
import torch
import re
import random
import time
from pathlib import Path
from PIL import PngImagePlugin
from datetime import datetime as dt
from dataclasses import dataclass
from csv import DictWriter
from repo.stable_diffusion.src import (
    args,
    Text2ImagePipeline,
    get_schedulers,
    set_init_device_flags,
)

# This has to come before importing cache objects
if args.clear_all:
    print("CLEARING ALL, EXPECT SEVERAL MINUTES TO RECOMPILE")
    from glob import glob
    import shutil

    vmfbs = glob(os.path.join(os.getcwd(), "*.vmfb"))
    for vmfb in vmfbs:
        if os.path.exists(vmfb):
            os.remove(vmfb)
    home = os.path.expanduser("~")
    if os.name == "nt":  # Windows
        appdata = os.getenv("LOCALAPPDATA")
        shutil.rmtree(os.path.join(appdata, "AMD/VkCache"), ignore_errors=True)
        shutil.rmtree(os.path.join(home, "shark_tank"), ignore_errors=True)
    elif os.name == "unix":
        shutil.rmtree(os.path.join(home, ".cache/AMD/VkCache"))
        shutil.rmtree(os.path.join(home, ".local/shark_tank"))


# save output images and the inputs correspoding to it.
def save_output_img(output_img):
    output_path = args.output_dir if args.output_dir else Path.cwd()
    generated_imgs_path = Path(output_path, "generated_imgs")
    generated_imgs_path.mkdir(parents=True, exist_ok=True)
    csv_path = Path(generated_imgs_path, "imgs_details.csv")

    prompt_slice = re.sub("[^a-zA-Z0-9]", "_", args.prompts[0][:15])
    out_img_name = (
        f"{prompt_slice}_{args.seed}_{dt.now().strftime('%y%m%d_%H%M%S')}"
    )
    out_img_path = Path(generated_imgs_path, f"{out_img_name}.jpg")

    if args.output_img_format == "jpg":
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.jpg")
        output_img.save(out_img_path, quality=95, subsampling=0)
    else:
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.png")
        pngInfo = PngImagePlugin.PngInfo()

        if args.write_metadata_to_png:
            pngInfo.add_text(
                "parameters",
                f"{args.prompts[0]}\nNegative prompt: {args.negative_prompts[0]}\nSteps:{args.steps}, Sampler: {args.scheduler}, CFG scale: {args.guidance_scale}, Seed: {args.seed}, Size: {args.width}x{args.height}, Model: {args.hf_model_id}",
            )

        output_img.save(
            output_path / f"{out_img_name}.png", "PNG", pnginfo=pngInfo
        )

        if args.output_img_format not in ["png", "jpg"]:
            print(
                f"[ERROR] Format {args.output_img_format} is not supported yet."
                "Image saved as png instead. Supported formats: png / jpg"
            )

    new_entry = {
        "VARIANT": args.hf_model_id,
        "SCHEDULER": args.scheduler,
        "PROMPT": args.prompts[0],
        "NEG_PROMPT": args.negative_prompts[0],
        "SEED": args.seed,
        "CFG_SCALE": args.guidance_scale,
        "PRECISION": args.precision,
        "STEPS": args.steps,
        "HEIGHT": args.height,
        "WIDTH": args.width,
        "MAX_LENGTH": args.max_length,
        "OUTPUT": out_img_path,
    }

    with open(csv_path, "a") as csv_obj:
        dictwriter_obj = DictWriter(csv_obj, fieldnames=list(new_entry.keys()))
        dictwriter_obj.writerow(new_entry)
        csv_obj.close()

    if args.save_metadata_to_json:
        del new_entry["OUTPUT"]
        with open(f"{output_path}/{out_img_name}.json", "w") as f:
            json.dump(new_entry, f, indent=4)


txt2img_obj = None
config_obj = None
schedulers = None

if __name__ == "__main__":
    dtype = torch.float32 if args.precision == "fp32" else torch.half
    cpu_scheduling = not args.scheduler.startswith("Shark")
    set_init_device_flags()
    schedulers = get_schedulers(args.hf_model_id)
    scheduler_obj = schedulers[args.scheduler]

    txt2img_obj = Text2ImagePipeline.from_pretrained(
        scheduler_obj,
        args.import_mlir,
        args.hf_model_id,
        args.ckpt_loc,
        args.precision,
        args.max_length,
        args.batch_size,
        args.height,
        args.width,
        args.use_base_vae,
        False
    )

    start_time = time.time()
    
    print(f"images count={args.total_count} \n")
    

    for num in range(args.total_count):
        if args.seed == -1:
            args.seed = random.randrange(9223372036854775807)
            print(f"Maked random seed at seed={args.seed} \n")
        else:             
            print(f"Generating at seed={args.seed} \n")

        generated_imgs = txt2img_obj.generate_images(
            args.prompts, args.negative_prompts,
            args.batch_size,
            args.height, args.width,
            args.steps,
            args.guidance_scale, args.seed,
            args.max_length, dtype,
            args.use_base_vae, cpu_scheduling,                    
        )
        args.seed = args.seed + 1
        save_output_img(generated_imgs[0])

    total_time = time.time() - start_time
    text_output = f"prompt={args.prompts}"
    text_output += f"\nnegative prompt={args.negative_prompts}"
    text_output += f"\nmodel_id={args.hf_model_id}, ckpt_loc={args.ckpt_loc}"
    text_output += f"\nscheduler={args.scheduler}, device={args.device}"
    text_output += f"\nsteps={args.steps}, guidance_scale={args.guidance_scale}, seed={args.seed}, size={args.height}x{args.width}"
    text_output += (
        f", batch size={args.batch_size}, max_length={args.max_length}"
    )
    text_output += txt2img_obj.log
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"
    print(text_output)
