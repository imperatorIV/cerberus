from PIL import Image
import os
import json
from transformers import CLIPProcessor, CLIPModel


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

caption_files = os.listdir("baseline_models/data/caption_jsons")
results = {}
for file in caption_files:
    path = os.path.join("baseline_models/data/caption_jsons", file)
    with open(path, "r") as f:
        data = json.load(f)
    num_correct, total = 0, 0
    for key in data.keys():
        img_path = os.path.join("baseline_models/data/images/val2017", data[key]["filename"])
        img = Image.open(img_path)
        inputs = processor(text=[data[key]["caption"], data[key]["negative_caption"]],
                           images=img, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        total += 1
        if probs[0, 0] >= probs[0, 1]:
            num_correct += 1
    results[file.split(".")[0]] = num_correct / total
    print(results[file.split(".")[0]])
json_res = json.dumps(results)
with open("CLIP-ViT-B32.json", "w") as outfile:
    outfile.write(json_res)
