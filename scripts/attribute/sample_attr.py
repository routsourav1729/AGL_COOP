import os
import argparse
import json
import numpy as np
import torch
import clip
from pathlib import Path
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
from PIL import Image
class AttributeSampler:
    """
    Attribute sampler for ArGue methodology, implementing clustering and
    similarity-based selection of attributes.
    """
    
    def __init__(self, clip_model="ViT-B/32", device=None):
        """
        Initialize the attribute sampler.
        
        Args:
            clip_model: Name of the CLIP model to use
            device: Device to run the model on (cpu or cuda)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading CLIP model {clip_model} on {self.device}...")
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        
    def compute_similarity(self, text: str, image_features: torch.Tensor) -> float:
        """
        Compute similarity between a text and an image in CLIP space.
        
        Args:
            text: Text attribute
            image_features: Image features from CLIP
            
        Returns:
            Similarity score
        """
        # Encode text with CLIP
        with torch.no_grad():
            text_inputs = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).item()
        return similarity
    
    def sample_attributes(self, 
                          attributes: List[str],
                          image_paths: List[str],
                          num_clusters: int = 3) -> List[str]:
        """
        Sample attributes using clustering and similarity-based selection.
        
        Args:
            attributes: List of attributes to sample from
            image_paths: List of image paths for the class
            num_clusters: Number of clusters to create
            
        Returns:
            List of selected attributes
        """
        if len(attributes) <= num_clusters:
            return attributes
            
        # 1. Encode attributes with CLIP
        with torch.no_grad():
            text_inputs = clip.tokenize(attributes).to(self.device)
            text_features = self.model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 2. Cluster attributes
        text_features_np = text_features.cpu().numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(text_features_np)
        
        # 3. Compute image features (average across images)
        image_features_list = []
        for img_path in image_paths[:min(len(image_paths), 10)]:  # Limit to 10 images for efficiency
            try:
                image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_feature = self.model.encode_image(image)
                    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
                image_features_list.append(image_feature)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        
        if not image_features_list:
            # If no images could be processed, return random attributes
            selected_indices = []
            for cluster_id in range(num_clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]
                if len(cluster_indices) > 0:
                    selected_indices.append(np.random.choice(cluster_indices))
            return [attributes[i] for i in selected_indices]
        
        image_features = torch.mean(torch.cat(image_features_list, dim=0), dim=0, keepdim=True)
        
        # 4. For each cluster, select the attribute with highest similarity to images
        selected_indices = []
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
                
            # Calculate similarity for each attribute in this cluster
            similarities = []
            for idx in cluster_indices:
                sim = (100.0 * image_features @ text_features[idx:idx+1].T).item()
                similarities.append((idx, sim))
            
            # Sort by similarity and select the attribute with highest similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            selected_indices.append(similarities[0][0])
        
        # Return selected attributes
        return [attributes[i] for i in selected_indices]
    
    def process_dataset(self, 
                        attributes_file: str,
                        dataset_path: str,
                        output_file: str,
                        num_clusters: int = 3) -> Dict[str, List[str]]:
        """
        Process a whole dataset, sampling attributes for each class.
        
        Args:
            attributes_file: Path to the JSON file with attributes
            dataset_path: Path to the dataset directory
            output_file: Path to save the sampled attributes
            num_clusters: Number of clusters to create for each class
            
        Returns:
            Dictionary mapping class names to sampled attributes
        """
        # Load attributes
        with open(attributes_file, 'r') as f:
            all_attributes = json.load(f)
            
        # Find image paths for each class - this is dataset-specific and may need adjustment
        class_to_images = self._find_images_by_class(dataset_path, list(all_attributes.keys()))
        
        sampled_attributes = {}
        total_classes = len(all_attributes)
        
        # Process each class
        for i, (class_name, attributes) in enumerate(all_attributes.items(), 1):
            print(f"Processing class {i}/{total_classes}: {class_name}")
            
            image_paths = class_to_images.get(class_name, [])
            if not image_paths:
                print(f"Warning: No images found for class {class_name}")
                # Fall back to just selecting random attributes
                if len(attributes) <= num_clusters:
                    sampled = attributes
                else:
                    sampled = np.random.choice(attributes, num_clusters, replace=False).tolist()
            else:
                sampled = self.sample_attributes(attributes, image_paths, num_clusters)
            
            sampled_attributes[class_name] = sampled
            print(f"Selected attributes: {sampled}")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(sampled_attributes, f, indent=2)
            
        return sampled_attributes
    
    def _find_images_by_class(self, dataset_path: str, class_names: List[str]) -> Dict[str, List[str]]:
        """
        Find image paths for each class in the dataset.
        This is a generic implementation that might need to be customized for specific datasets.
        
        Args:
            dataset_path: Path to the dataset directory
            class_names: List of class names to find images for
            
        Returns:
            Dictionary mapping class names to lists of image paths
        """
        # Try to guess the dataset structure
        dataset_path = Path(dataset_path)
        result = {}
        
        # Try finding images in common directory structures
        for class_name in class_names:
            # Look for directories matching the class name (with variations)
            class_dir_candidates = [
                dataset_path / "images" / class_name,
                dataset_path / class_name,
                dataset_path / class_name.lower().replace(" ", "_"),
                dataset_path / "jpg" / class_name.lower().replace(" ", "_")
            ]
            
            for class_dir in class_dir_candidates:
                if class_dir.exists() and class_dir.is_dir():
                    # Find image files with common extensions
                    image_paths = []
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        image_paths.extend([str(p) for p in class_dir.glob(ext)])
                    
                    if image_paths:
                        result[class_name] = image_paths
                        break
        
        return result
        

def main(args):
    from PIL import Image
    
    # Initialize attribute sampler
    sampler = AttributeSampler(clip_model=args.clip_model, device=args.device)
    
    # Process each dataset
    for attributes_file in args.attributes_files:
        attributes_path = Path(attributes_file)
        dataset_name = attributes_path.stem.replace("_attributes", "")
        
        print(f"\n=== Processing {dataset_name} ===")
        
        # Determine dataset path and output file
        dataset_path = Path(args.data_dir) / dataset_name
        output_file = Path(args.output_dir) / f"{dataset_name}_sampled_attributes.json"
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process dataset
        sampler.process_dataset(
            attributes_file=attributes_file,
            dataset_path=dataset_path,
            output_file=output_file,
            num_clusters=args.num_clusters
        )
        
        print(f"Saved sampled attributes to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample attributes from a pool using ArGue methodology")
    parser.add_argument(
        "--attributes_files",
        type=str,
        nargs="+",
        required=True,
        help="Path to JSON files containing attributes for each dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root directory containing all datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sampled_attributes",
        help="Directory to save sampled attributes"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "RN50", "RN101"],
        help="CLIP model to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run CLIP on (default: use CUDA if available, otherwise CPU)"
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=3,
        help="Number of clusters to create for attribute sampling"
    )
    
    args = parser.parse_args()
    main(args)