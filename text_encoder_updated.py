import spacy
from transformers import CLIPTextModel, CLIPTokenizer
import torch

class TextEncoder:
    def __init__(self):
        # Load spaCy and CLIP models
        self.nlp = spacy.load('en_core_web_trf')
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def extract_features(self, text):
        doc = self.nlp(text)
        features = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:  # Nouns and proper nouns
                subtree = [child for child in token.subtree if child.dep_ in ['amod', 'compound']]
                feature_phrase = ' '.join([node.text for node in subtree] + [token.text])
                features.append(feature_phrase)
        return features

    def segment_text(self, features):
        segments = []
        for feature in features:
            doc = self.nlp(feature)
            segment = ' '.join([token.text for token in doc])
            segments.append(segment)
        return segments

    def encode_text(self, segments):
        encoded_features = []
        max_length = 0
        max_length_segment = ""

        # Tokenize all segments and find the maximum length
        tokenized_outputs = [self.clip_tokenizer(segment, return_tensors="pt", padding=True, truncation=True) for segment in segments]
        for segment, output in zip(segments, tokenized_outputs):
            current_length = output.input_ids.size(1)
            if current_length > max_length:
                max_length = current_length
                max_length_segment = segment

        # Pad and encode all tokenized outputs
        for output in tokenized_outputs:
            input_ids = output.input_ids
            attention_mask = output.attention_mask
            pad_length = max_length - input_ids.size(1)
            if pad_length > 0:
                padded_ids = torch.cat([input_ids, torch.zeros((1, pad_length), dtype=torch.int64)], dim=1)
                padded_mask = torch.cat([attention_mask, torch.zeros((1, pad_length), dtype=torch.int64)], dim=1)
            else:
                padded_ids = input_ids
                padded_mask = attention_mask
            outputs = self.clip_model(input_ids=padded_ids, attention_mask=padded_mask)
            encoded_features.append(outputs.last_hidden_state.squeeze(0))
        return torch.stack(encoded_features)

    def text_to_encoded_features(self, input_text):
        features = self.extract_features(input_text)
        segments = self.segment_text(features)
        encoded_features = self.encode_text(segments)
        pooled_features, _ = torch.max(encoded_features, dim=1)
        final_feature_vector = torch.mean(pooled_features, dim=0)
        return final_feature_vector

# Main execution
if __name__ == "__main__":
    encoder = TextEncoder()
    sample_text = "The quick brown fox jumps over the lazy dog near the peaceful river under the bright sky."
    encoded_output = encoder.text_to_encoded_features(sample_text)
    print(encoded_output)