import argparse
import json
from pathlib import Path
import imghdr

from PIL import Image
import torch
import clip
from model.ZeroCLIP import CLIPTextGenerator

parser = argparse.ArgumentParser(description="Generate image captions.")
parser.add_argument("--path", default='../test_images/fish')
parser.add_argument("--cond_text", type=str, default="Image of a")
parser.add_argument("--beam_size", type=int, default=5)
args = parser.parse_args()

image_path = Path(args.path)
image_paths = [x for x in image_path.iterdir() if x.is_file() and imghdr.what(x) is not None]
captions_all = []
for path in image_paths:
    text_generator = CLIPTextGenerator(**vars(args))

    image_features = text_generator.get_img_feature([path], None)
    captions = text_generator.run(image_features, cond_text=args.cond_text, beam_size=args.beam_size)

    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)
    caption = captions[best_clip_idx]
    captions_all.append(caption)
    print(f"{path}: {caption}")


with open(Path(image_path, "captions.json"), 'w', encoding="utf8") as outfile:
    json.dump({"paths": [x.name for x in image_paths], "captions": captions_all}, outfile)
