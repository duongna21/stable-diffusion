import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import os
from tqdm import tqdm

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

i = 0
# load sample image
with open('blip_captions.csv', 'w') as w:
    for img in tqdm(os.listdir("ffhq-512")):
        raw_image = Image.open(f"ffhq-512/{img}").convert("RGB")
        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        # generate caption
        pred = model.generate({"image": image})
        w.write(f"\n{img},{pred[0]}")
        # ['a large fountain spewing water into the air']
