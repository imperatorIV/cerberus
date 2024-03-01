"""
We use the same strategy as the author to display visualizations
as in the examples shown in the paper. The strategy used is adapted for
PyTorch from here:
https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
"""

import argparse, json, os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.transform
import torch
import torchvision.transforms as transforms
from math import ceil
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
from rouge import Rouge
from pycocoevalcap.rouge.rouge import Rouge
from tqdm import tqdm  # Import tqdm for progress bar

# # Ensure NLTK 'punkt' package is downloaded
# nltk.download('punkt')

from dataset import ImageCaptionDataset, pil_loader
from decoder import Decoder
from encoder import Encoder
from train import data_transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def load_image_paths(json_file):
    with open(json_file, 'r') as f:
        image_paths = json.load(f)
    return image_paths

def load_captions_json(json_file):
    with open(json_file, 'r') as f:
        captions_data = json.load(f)
    return captions_data

def generate_caption(encoder, decoder, img_path, word_dict):
    img = pil_loader(img_path)
    img_transform = Compose([Resize((224, 224)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = img_transform(img)
    img = torch.unsqueeze(img, 0)  # Add batch dimension

    img_features = encoder(img)
    sentence, _ = decoder.caption(img_features, beam_size=1)

    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens = [token_dict[word_idx] for word_idx in sentence if word_idx in token_dict and token_dict[word_idx] not in ['<pad>', '<eos>']]

    return ' '.join(sentence_tokens)

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    # Remove punctuation, non-alphabetic tokens, and <pad> tokens
    tokens = [token for token in tokens if token.isalpha() and token != '<pad>']
    # Reconstruct the text from tokens
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


def preprocess_references_and_candidates(captions_data, word_dict):
    preprocessed_references = []
    for caption_ids_list in captions_data:
        # Convert each list of caption IDs to a single string caption
        caption = ' '.join([word_dict[str(id)] for id in caption_ids_list if str(id) in word_dict])
        # Preprocess the caption
        preprocessed_caption = preprocess_text(caption)
        # Append the preprocessed caption to the list of references
        preprocessed_references.append([preprocessed_caption])
    return preprocessed_references

def evaluate_model_on_coco(encoder, decoder, coco_data, val_captions_data, word_dict):
    references = preprocess_references_and_candidates(val_captions_data, word_dict)
    candidates = []

    # Reverse the word_dict to map indices back to words
    token_dict = {v: k for k, v in word_dict.items()}

    for img_path, _ in tqdm(coco_data, desc="Evaluating model"):
        # Generate and prepare candidate caption
        generated_caption = generate_caption(encoder, decoder, img_path, word_dict)
        cleaned_generated_caption = clean_caption(generated_caption)
        candidates.append([cleaned_generated_caption])  # Ensure candidates are in a list of lists for BLEU calculation

    # Preprocess candidates
    preprocessed_candidates = [[preprocess_text(can[0])] for can in candidates]

    # Calculate BLEU score
    bleu_score = corpus_bleu(references, preprocessed_candidates)
    print(f"BLEU score: {bleu_score}")

def clean_caption(caption):
    # Remove special tokens and tokenize
    tokens = word_tokenize(caption)
    cleaned_tokens = [token for token in tokens if token not in ['<start>', '<eos>', '<pad>']]
    return ' '.join(cleaned_tokens)  # Return a string of cleaned caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell Caption Generator')
    parser.add_argument('--img-path', type=str, help='path to image')
    parser.add_argument('--network', choices=['vgg19', 'resnet152'], default='vgg19', help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--model', type=str, help='path to model parameters')
    parser.add_argument('--data-path', type=str, default='data/coco/imgs/val2017', help='path to image data')
    parser.add_argument('--caption-path', type=str, default='captions_data', help='path to caption JSON files')
    parser.add_argument('--img-paths-json', type=str, default='data/coco/val_img_paths.json', help='path to validation image paths JSON file (default: val_img_paths.json)')

    args = parser.parse_args()

    # Load word dictionary
    with open("data/coco/word_dict.json", 'r') as f:
        word_dict = json.load(f)
    vocabulary_size = len(word_dict)

    # Load encoder and decoder
    encoder = Encoder('vgg19')  # Assuming you're using 'vgg19'
    decoder = Decoder(vocabulary_size, encoder.dim)
    decoder.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))


    scores = {}  # Store scores for each JSON file

    # Iterate over all JSON files containing captions
    for file_name in tqdm([f for f in os.listdir(args.caption_path) if f.endswith('.json')], desc="Processing JSON Files"):
        file_path = os.path.join(args.caption_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)

        cumulative_similarity_score = 0
        total_images = 0

        # Iterate over images in the JSON file
        for img_key, img_data in captions_data.items():
            img_path = os.path.join(args.data_path, img_data["filename"])

            # Load positive and negative captions
            positive_caption = preprocess_text(img_data["caption"])
            negative_caption = preprocess_text(img_data["negative_caption"])

            # Generate caption for the image
            generated_caption = preprocess_text(generate_caption(encoder, decoder, img_path, word_dict))

            # Remove <start> and <eos> tokens
            generated_caption = generated_caption.replace('<start>', '').replace('<eos>', '')

            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

            # Calculate Rouge scores
            positive_scores = scorer.score(generated_caption, positive_caption)
            negative_scores = scorer.score(generated_caption, negative_caption)

            # Calculate similarity scores using ROUGE score for demonstration purposes
            positive_similarity_score = positive_scores['rougeL'].fmeasure
            negative_similarity_score = negative_scores['rougeL'].fmeasure

            # Update cumulative similarity score
            cumulative_similarity_score += positive_similarity_score - negative_similarity_score

            total_images += 1

        # Calculate average similarity score
        average_similarity_score = cumulative_similarity_score / total_images if total_images != 0 else 0

        # Store the average similarity score for the current JSON file
        scores[file_name.split('.')[0]] = average_similarity_score

    # Print scores in the desired format
    print(json.dumps(scores))