import os
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Import OpenAI for attribute generation
try:
    import openai
except ImportError:
    print("OpenAI package not installed. Attribute generation will not work.")

# Import CLIP
try:
    from clip import clip
    from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
except ImportError:
    print("CLIP package not installed. Attribute sampling will not work.")

class AttributeManager:
    """
    Combined class for generating and sampling attributes for vision-language models
    following ArGue paper methodology.
    """
    
    def __init__(
        self, 
        save_dir: str = "attributes",
        api_key: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the attribute manager.
        
        Args:
            save_dir: Directory to save generated attributes
            api_key: OpenAI API key (optional for sampling-only usage)
            verbose: Whether to print progress information
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Set up OpenAI API if key is provided
        self.api_key = api_key
        if api_key:
            try:
                openai.api_key = api_key
            except NameError:
                print("OpenAI module not available. Install with 'pip install openai'")
        
        # Templates from ArGue paper for attribute generation
        self.templates = [
            {
                "system": "You are an expert at describing visual attributes of objects in images.",
                "user": "Describe what a {type} {class_name} looks like in a photo, list {num} pieces? Each piece should be a short phrase under 5 words."
            },
            {
                "system": "You are an expert at describing visual attributes of objects in images.",
                "user": "Visually describe a {class_name}, a type of {type}, list {num} pieces? Each piece should be a short phrase under 5 words."
            },
            {
                "system": "You are an expert at describing visual attributes of objects in images.", 
                "user": "How to distinguish a {class_name} which is a {type}, list {num} pieces? Each piece should be a short phrase under 5 words."
            }
        ]
        
        # Example prompts for in-context learning
        self.examples = [
            {
                "user": "Describe what an animal giraffe looks like in a photo, list 5 pieces? Each piece should be a short phrase under 5 words.",
                "assistant": "There are 5 useful visual features for a giraffe in a photo:\n- spotted coat pattern\n- extremely long neck\n- tall slender legs\n- small head\n- short tan-colored horns"
            }
        ]
        
        # Initialize CLIP model (lazy loading)
        self._clip_model = None
        self._clip_preprocess = None
        self._tokenizer = None

    def _load_clip_model(self):
        """Load CLIP model if it hasn't been loaded yet"""
        if self._clip_model is None:
            if self.verbose:
                print("Loading CLIP model...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = 'ViT-B/32'
            model_path = clip._download(clip._MODELS[model_name])
            
            try:
                model = torch.jit.load(model_path, map_location=device).eval()
                state_dict = None
            except RuntimeError:
                state_dict = torch.load(model_path, map_location=device)
                
            self._clip_model = clip.build_model(state_dict or model.state_dict())
            self._clip_model = self._clip_model.to(device)
            self._clip_preprocess = clip._transform(self._clip_model.visual.input_resolution)
            self._tokenizer = _Tokenizer()
            
            if self.verbose:
                print(f"CLIP model loaded on {device}")
    
    def generate_class_attributes(
        self, 
        class_name: str, 
        type_name: str,
        num_attributes: int = 5,
        temperature: float = 0.8
    ) -> List[str]:
        """
        Generate attributes for a given class using all templates.
        
        Args:
            class_name: Name of the class
            type_name: Type of object (e.g., "flower", "animal")
            num_attributes: Number of attributes to generate per template
            temperature: Temperature for generation
            
        Returns:
            List of generated attributes
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is required for attribute generation")
            
        all_attributes = []
        
        # Generate attributes using each template
        for template in self.templates:
            try:
                messages = [
                    {"role": "system", "content": template["system"]},
                    # Add examples for in-context learning
                    {"role": "user", "content": self.examples[0]["user"]},
                    {"role": "assistant", "content": self.examples[0]["assistant"]},
                    # Add actual query
                    {"role": "user", "content": template["user"].format(
                        class_name=class_name,
                        type=type_name,
                        num=num_attributes
                    )}
                ]
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=200
                )
                
                # Parse attributes from response
                attributes = self._parse_response(response.choices[0].message["content"])
                all_attributes.extend(attributes)
                
            except Exception as e:
                print(f"Error generating attributes for {class_name} with template: {e}")
                continue
                
        # Remove duplicates while preserving order
        seen = set()
        unique_attributes = [
            attr for attr in all_attributes 
            if not (attr in seen or seen.add(attr))
        ]
        
        return unique_attributes

    def _parse_response(self, response: str) -> List[str]:
        """Parse LLM response into clean list of attributes."""
        attributes = []
        for line in response.split('\n'):
            line = line.strip()
            # Match lines starting with common list markers
            if line and (line.startswith('-') or line.startswith('•')):
                # Clean up the attribute text
                attr = line.lstrip('-•').strip()
                # Remove any leading numbers or dots
                attr = ' '.join(attr.split()[0:]).strip()
                if attr:
                    attributes.append(attr)
        return attributes

    def generate_and_save_dataset_attributes(
        self,
        classnames: List[str],
        type_name: str,
        dataset_name: str,
        num_attributes: int = 5
    ) -> Dict[str, List[str]]:
        """
        Generate and save attributes for all classes in a dataset.
        
        Args:
            classnames: List of class names
            type_name: Type of object (e.g., "flower")
            dataset_name: Name of the dataset for saving
            num_attributes: Number of attributes to generate per template
            
        Returns:
            Dictionary mapping class names to attributes
        """
        all_class_attributes = {}
        
        for class_name in tqdm(classnames, desc="Generating attributes"):
            # Generate attributes for current class
            attributes = self.generate_class_attributes(
                class_name=class_name,
                type_name=type_name,
                num_attributes=num_attributes
            )
            all_class_attributes[class_name] = attributes
            
        # Save results
        save_path = self.save_dir / f"{dataset_name}_attributes.json"
        with open(save_path, 'w') as f:
            json.dump(all_class_attributes, f, indent=2)
            
        if self.verbose:
            print(f"Saved attributes to {save_path}")
            
        return all_class_attributes

    def load_dataset_attributes(self, dataset_name: str) -> Optional[Dict[str, List[str]]]:
        """Load previously generated attributes for a dataset."""
        path = self.save_dir / f"{dataset_name}_attributes.json"
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def get_image_embeddings(self, image_paths: List[str]) -> torch.Tensor:
        """Get CLIP embeddings for images"""
        self._load_clip_model()  # Ensure model is loaded
        
        device = next(self._clip_model.parameters()).device
        batch_size = 32  # Process images in batches to avoid OOM
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(self._clip_preprocess(img))
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    # Use a blank image as fallback
                    blank = Image.new('RGB', (224, 224), color='gray')
                    images.append(self._clip_preprocess(blank))
            
            image_batch = torch.stack(images).to(device)
            
            with torch.no_grad():
                batch_features = self._clip_model.encode_image(image_batch)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                all_features.append(batch_features)
        
        # Concatenate all batches
        return torch.cat(all_features, dim=0)

    def get_text_embeddings(self, attributes: List[str], classname: str) -> torch.Tensor:
        """Get CLIP embeddings for attributes"""
        self._load_clip_model()  # Ensure model is loaded
        
        device = next(self._clip_model.parameters()).device
        text_prompts = [f"a photo of a {classname} which has {attr}" for attr in attributes]
        text_tokens = clip.tokenize(text_prompts).to(device)
        
        with torch.no_grad():
            text_features = self._clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def sample_attributes(
        self,
        class_attributes: Dict[str, List[str]],
        class_images: Dict[str, List[str]],
        num_clusters: int = 3,
        max_images_per_class: int = 10
    ) -> Dict[str, List[str]]:
        """
        Sample diverse and representative attributes using clustering
        
        Args:
            class_attributes: Dictionary mapping class names to attributes
            class_images: Dictionary mapping class names to image paths
            num_clusters: Number of attribute clusters to create
            max_images_per_class: Maximum number of images to use per class
            
        Returns:
            Dictionary mapping class names to sampled attributes
        """
        self._load_clip_model()  # Ensure model is loaded
        sampled_attributes = {}
        
        for classname, attributes in tqdm(class_attributes.items(), desc="Sampling attributes"):
            if classname not in class_images or not attributes:
                continue
                
            # Limit number of images to process
            image_paths = class_images[classname][:max_images_per_class]
            if not image_paths:
                continue
                
            # Get image embeddings
            try:
                image_features = self.get_image_embeddings(image_paths)
                
                # Get text embeddings
                text_features = self.get_text_embeddings(attributes, classname)
                
                # Use fewer clusters if we have fewer attributes than requested clusters
                n_clusters = min(num_clusters, len(attributes))
                
                # Cluster text embeddings
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(text_features.cpu().numpy())
                
                # Select best attribute from each cluster
                selected_attributes = []
                for cluster_idx in range(n_clusters):
                    cluster_mask = clusters == cluster_idx
                    # Skip empty clusters
                    if not any(cluster_mask):
                        continue
                        
                    cluster_features = text_features[cluster_mask]
                    cluster_attributes = [attr for attr, mask in zip(attributes, cluster_mask) if mask]
                    
                    # Calculate similarity with images
                    similarities = (100.0 * image_features @ cluster_features.T).mean(dim=0)
                    best_idx = similarities.argmax().item()
                    selected_attributes.append(cluster_attributes[best_idx])
                
                sampled_attributes[classname] = selected_attributes
                
            except Exception as e:
                print(f"Error sampling attributes for {classname}: {e}")
                continue
        
        return sampled_attributes
        
    def sample_and_save_attributes(
        self,
        dataset_name: str,
        class_images: Dict[str, List[str]],
        num_clusters: int = 3,
        attribute_dict: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, List[str]]:
        """
        Sample and save attributes for a dataset
        
        Args:
            dataset_name: Name of the dataset
            class_images: Dictionary mapping class names to image paths
            num_clusters: Number of attribute clusters to create
            attribute_dict: Optional pre-loaded attribute dictionary
            
        Returns:
            Dictionary mapping class names to sampled attributes
        """
        # Load attributes if not provided
        if attribute_dict is None:
            attribute_dict = self.load_dataset_attributes(dataset_name)
            
        if not attribute_dict:
            raise ValueError(f"No attributes found for dataset {dataset_name}")
            
        # Sample attributes
        sampled_attributes = self.sample_attributes(
            class_attributes=attribute_dict,
            class_images=class_images,
            num_clusters=num_clusters
        )
        
        # Save results
        save_path = self.save_dir / f"{dataset_name}_sampled_attributes.json"
        with open(save_path, 'w') as f:
            json.dump(sampled_attributes, f, indent=2)
            
        if self.verbose:
            print(f"Saved sampled attributes to {save_path}")
            
        return sampled_attributes

def process_dataset(
    dataset_config: Dict,
    api_key: Optional[str] = None,
    generate_attrs: bool = True,
    sample_attrs: bool = True
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Process a dataset to generate and sample attributes
    
    Args:
        dataset_config: Configuration for the dataset
        api_key: OpenAI API key (required for generation)
        generate_attrs: Whether to generate attributes
        sample_attrs: Whether to sample attributes
        
    Returns:
        Tuple of (generated attributes, sampled attributes)
    """
    # Extract configuration
    dataset_name = dataset_config["name"]
    data_dir = dataset_config["data_dir"]
    classnames = dataset_config["classnames"]
    type_name = dataset_config["type"]
    image_root = dataset_config.get("image_root", "")
    class_images = dataset_config.get("class_images", {})
    split_file = dataset_config.get("split_file", "")
    attributes_dir = os.path.join(data_dir, "attributes")
    
    # Create attribute manager
    manager = AttributeManager(
        save_dir=attributes_dir,
        api_key=api_key
    )
    
    generated_attributes = None
    sampled_attributes = None
    
    # Generate attributes if requested
    if generate_attrs:
        if not api_key:
            raise ValueError("OpenAI API key is required for attribute generation")
            
        generated_attributes = manager.generate_and_save_dataset_attributes(
            classnames=classnames,
            type_name=type_name,
            dataset_name=dataset_name
        )
    
    # Sample attributes if requested
    if sample_attrs:
        # Load attributes if we didn't generate them
        if not generated_attributes:
            generated_attributes = manager.load_dataset_attributes(dataset_name)
            
        if not generated_attributes:
            print(f"No attributes found for {dataset_name}, skipping sampling")
            return generated_attributes, sampled_attributes
            
        # If class_images wasn't provided, try to build it from split file
        if not class_images and split_file and os.path.exists(split_file):
            class_images = {}
            try:
                with open(split_file, 'r') as f:
                    split_data = json.load(f)
                
                train_images = split_data.get('train', [])
                for item in train_images:
                    if len(item) == 3:  # [image_name, label, class_name]
                        image_name, _, classname = item
                        if classname not in class_images:
                            class_images[classname] = []
                        image_path = os.path.join(image_root, image_name)
                        class_images[classname].append(image_path)
            except Exception as e:
                print(f"Error loading split file: {e}")
        
        if not class_images:
            print(f"No class images found for {dataset_name}, skipping sampling")
            return generated_attributes, sampled_attributes
            
        # Sample attributes
        sampled_attributes = manager.sample_and_save_attributes(
            dataset_name=dataset_name,
            class_images=class_images,
            attribute_dict=generated_attributes
        )
    
    return generated_attributes, sampled_attributes


# Utility function for reading Oxford Flowers class names
def read_flowers_classnames(json_path: str) -> List[str]:
    """Read class names from Oxford Flowers dataset."""
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    # Convert from {'1': 'pink primrose', '2': 'hard-leaved pocket orchid'} format
    classnames = [name.strip() for name in cat_to_name.values()]
    return classnames