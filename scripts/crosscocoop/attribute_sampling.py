import os
import json
import torch
from pathlib import Path
import argparse
from sklearn.cluster import KMeans
from PIL import Image
from tqdm import tqdm
from dassl.utils import read_json

# Import from local CLIP module
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-file', type=str, 
                       default='/home/sourav/ALL_FILES/Thesis/attribute-guided-learning/data/oxford_flowers/split_zhou_OxfordFlowers.json')
    parser.add_argument('--attr-file', type=str,
                       default='/home/sourav/ALL_FILES/Thesis/attribute-guided-learning/data/oxford_flowers/attributes/oxford_flowers_attributes.json')
    parser.add_argument('--image-root', type=str,
                       default='/home/sourav/ALL_FILES/Thesis/attribute-guided-learning/data/oxford_flowers/jpg')
    parser.add_argument('--output-file', type=str,
                       default='data/oxford_flowers/attributes/sampled_attributes.json')
    parser.add_argument('--num-clusters', type=int, default=3)
    return parser.parse_args()

def load_clip_model():
    """Load CLIP model using CoOp's implementation"""
    model_path = clip._download(clip._MODELS['ViT-B/32'])
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

def get_image_embeddings(model, image_paths, preprocess):
    """Get CLIP embeddings for images"""
    images = torch.stack([preprocess(Image.open(p)) for p in image_paths])
    with torch.no_grad():
        image_features = model.encode_image(images.cuda())
    return image_features / image_features.norm(dim=-1, keepdim=True)

def get_text_embeddings(model, attributes, classname):
    """Get CLIP embeddings for attributes"""
    tokenizer = _Tokenizer()
    text_prompts = [f"a photo of a {classname} which has {attr}" for attr in attributes]
    text_tokens = clip.tokenize(text_prompts).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model
    model = load_clip_model()
    model = model.cuda()
    model.eval()
    
    # Load dataset split info
    split_data = read_json(args.split_file)
    train_images = split_data['train']
    
    # Load attributes
    with open(args.attr_file, 'r') as f:
        class_attributes = json.load(f)
    
    # Group images by class
    class_to_images = {}
    for item in train_images:
        image_name, _, classname = item  # Unpack [image_name, label, class_name]
        if classname not in class_to_images:
            class_to_images[classname] = []
        class_to_images[classname].append(os.path.join(args.image_root, image_name))
    
    sampled_attributes = {}
    preprocess = clip._transform(model.visual.input_resolution)
    
    for classname, image_paths in tqdm(class_to_images.items()):
        if classname not in class_attributes:
            continue
            
        # Get image embeddings
        image_features = get_image_embeddings(model, image_paths[:10], preprocess)
        
        # Get text embeddings
        attributes = class_attributes[classname]
        text_features = get_text_embeddings(model, attributes, classname)
        
        # Cluster text embeddings
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=42)
        clusters = kmeans.fit_predict(text_features.cpu().numpy())
        
        # Select best attribute from each cluster
        selected_attributes = []
        for cluster_idx in range(args.num_clusters):
            cluster_mask = clusters == cluster_idx
            cluster_features = text_features[cluster_mask]
            cluster_attributes = [attr for attr, mask in zip(attributes, cluster_mask) if mask]
            
            # Calculate similarity with images
            similarities = (100.0 * image_features @ cluster_features.T).mean(dim=0)
            best_idx = similarities.argmax().item()
            selected_attributes.append(cluster_attributes[best_idx])
        
        sampled_attributes[classname] = selected_attributes
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(sampled_attributes, f, indent=2)

if __name__ == '__main__':
    main()