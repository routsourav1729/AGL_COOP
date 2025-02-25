#!/usr/bin/env python3
import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from PIL import Image
from dassl.utils import read_json
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import openai

# -------------------------------------------------------------------
# Common Class: AttributeGenerator
# -------------------------------------------------------------------
class AttributeGenerator:
    """Generates visual attributes using LLMs with discriminative prompts."""
    def __init__(self, api_key: str, save_dir: str = "attributes", max_tokens: int = 50):
        self.api_key = api_key
        openai.api_key = api_key
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        # Discriminative templates for generating attributes
        self.templates = [
            {
                "system": ("You are an expert at describing visual attributes of objects in images. "
                           "Keep each attribute under 50 tokens."),
                "user": ("Q: Provide 5 short visual attributes of a {class_name} (flower), each <50 tokens.\nA:")
            },
            {
                "system": ("You are an expert at describing visual attributes of objects in images. "
                           "Keep each attribute under 50 tokens."),
                "user": ("Q: Give 5 brief traits that differentiate a {class_name} from similar flowers, "
                         "each <50 tokens.\nA:")
            },
            {
                "system": ("You are an expert at describing visual attributes of objects in images. "
                           "Keep each attribute under 50 tokens."),
                "user": ("Q: Describe 5 distinct indicators that help identify a {class_name} (flower), "
                         "each <50 tokens.\nA:")
            }
        ]
        # Optional in-context learning example
        self.examples = [
            {
                "user": "Describe what an animal giraffe looks like in a photo, list 6 pieces?",
                "assistant": ("There are 6 useful visual features for a giraffe in a photo:\n"
                              "- covered with a spotted coat\n"
                              "- has a short, stocky body\n"
                              "- has a long neck\n"
                              "- owns a small neck to its body\n"
                              "- is yellow or brown in color\n"
                              "- has a black tufted tail")
            }
        ]

    def generate_class_attributes(self, class_name: str, type_name: str, num_attributes: int = 5, temperature: float = 0.8) -> list:
        all_attributes = []
        for template in self.templates:
            try:
                messages = [
                    {"role": "system", "content": template["system"]},
                    {"role": "user", "content": self.examples[0]["user"]},
                    {"role": "assistant", "content": self.examples[0]["assistant"]},
                    {"role": "user", "content": template["user"].format(class_name=class_name, type=type_name, num=num_attributes)}
                ]
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens
                )
                attributes = self._parse_response(response.choices[0].message["content"])
                all_attributes.extend(attributes)
            except Exception as e:
                print(f"Error generating attributes for {class_name}: {e}")
                continue
        # Remove duplicates while preserving order
        seen = set()
        unique_attributes = [attr for attr in all_attributes if attr not in seen and not seen.add(attr)]
        return unique_attributes

    def _parse_response(self, response: str) -> list:
        attributes = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•')):
                attr = line.lstrip('-•').strip()
                attr = ' '.join(attr.split())
                if attr:
                    attributes.append(attr)
        return attributes

    def generate_and_save_dataset_attributes(self, classnames: list, type_name: str, dataset_name: str) -> dict:
        all_class_attributes = {}
        for class_name in classnames:
            attributes = self.generate_class_attributes(class_name=class_name, type_name=type_name)
            all_class_attributes[class_name] = attributes
        save_path = self.save_dir / f"{dataset_name}_attributes.json"
        with open(save_path, 'w') as f:
            json.dump(all_class_attributes, f, indent=2)
        return all_class_attributes

# -------------------------------------------------------------------
# Common Class: AttributeSampler
# -------------------------------------------------------------------
class AttributeSampler:
    """
    Samples discriminative attributes per class using CLIP embeddings and clustering.
    """
    def __init__(self, split_file: str, attr_file: str, image_root: str, output_file: str, num_clusters: int = 3, device: str = None):
        self.split_file = split_file
        self.attr_file = attr_file
        self.image_root = image_root
        self.output_file = output_file
        self.num_clusters = num_clusters
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_clip_model().to(self.device)
        self.model.eval()

    def load_clip_model(self):
        model_path = clip._download(clip._MODELS['ViT-B/32'])
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        model = clip.build_model(state_dict or model.state_dict())
        return model

    def get_image_embeddings(self, image_paths, preprocess):
        images = torch.stack([preprocess(Image.open(p)) for p in image_paths])
        with torch.no_grad():
            image_features = self.model.encode_image(images.to(self.device))
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def get_text_embeddings(self, attributes, classname):
        text_prompts = [f"a photo of a {classname} which has {attr}" for attr in attributes]
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def sample_attributes(self):
        split_data = read_json(self.split_file)
        train_images = split_data['train']
        with open(self.attr_file, 'r') as f:
            class_attributes = json.load(f)
        # Group images by class
        class_to_images = {}
        for item in train_images:
            image_name, _, classname = item  # Expecting item = [image_name, label, class_name]
            class_to_images.setdefault(classname, []).append(os.path.join(self.image_root, image_name))
        sampled_attributes = {}
        preprocess = clip._transform(self.model.visual.input_resolution)
        for classname, image_paths in tqdm(class_to_images.items(), desc="Sampling per class"):
            if classname not in class_attributes:
                continue
            image_features = self.get_image_embeddings(image_paths[:10], preprocess)
            attributes = class_attributes[classname]
            text_features = self.get_text_embeddings(attributes, classname)
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            clusters = kmeans.fit_predict(text_features.cpu().numpy())
            selected_attributes = []
            for cluster_idx in range(self.num_clusters):
                cluster_mask = clusters == cluster_idx
                if sum(cluster_mask) == 0:
                    continue
                cluster_features = text_features[cluster_mask]
                cluster_attributes = [attr for attr, mask in zip(attributes, cluster_mask) if mask]
                similarities = (100.0 * image_features @ cluster_features.T).mean(dim=0)
                best_idx = similarities.argmax().item()
                selected_attributes.append(cluster_attributes[best_idx])
            sampled_attributes[classname] = selected_attributes
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(sampled_attributes, f, indent=2)
        return sampled_attributes

