from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
import os


model_name = "bipin/image-caption-generator"

# load model
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

caption_files = os.listdir("baseline_models/data/caption_jsons")
results = {}
max_length = 128
num_beams = 4
for file in caption_files:
    path = os.path.join("baseline_models/data/caption_jsons", file)
    with open(path, "r") as f:
        data = json.load(f)
    num_correct, total = 0, 0
    for key in data.keys():
     
        total += 1
        img_path = os.path.join("baseline_models/data/images/val2017", data[key]["filename"])
        img = Image.open(img_path)
        if img.mode != 'RGB':
          img = img.convert(mode="RGB")

        pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, num_beams=num_beams, max_length=max_length)
        preds = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        bleu_score_positive = sentence_bleu([data[key]["caption"]], preds)
        bleu_score_negative = sentence_bleu([data[key]["negative_caption"]], preds)
    
        if bleu_score_positive >= bleu_score_negative :
          num_correct += 1 
    
    results[file.split(".")[0]] = num_correct / total
    print(results[file.split(".")[0]])

json_res = json.dumps(results)
with open("image-caption-generator.json", "w") as outfile:
    outfile.write(json_res)
