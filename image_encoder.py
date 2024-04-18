import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import json
import os

class ImageEncoder:

    def __init__(self, processor_link, model_link):

        self.processor = ViTImageProcessor.from_pretrained(processor_link)
        self.encoder = ViTModel.from_pretrained(model_link)
    

    def encode(self, file_path, data_path):

        with open(file_path, "r") as data_file:
            data = json.load(data_file)
        
        self.embeddings = {}
        
        for key, _ in data.items():
            img_file_name = data[key]["filename"]
            img_file_path = os.path.join(data_path, img_file_name)
            img = Image.open(img_file_path)
            input = self.processor(images=img, return_tensors="pt")
            output = self.encoder(**input)
            pooler_output = output.pooler_output
            self.embeddings[img_file_name] = pooler_output[0].tolist()
    

    def save_embeddings(self, file_path):
        
        with open(file_path, "w") as data_file:
            json.dump(self.embeddings, data_file)
