import os
import sys
import argparse
import json
import glob
import re
import scipy.io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
utils_path = os.path.abspath('/raid/biplab/souravr/thesis/multimodal/AGL')
sys.path.insert(0, utils_path)

from utils.attribute_gen import AttributeGenerator

class DatasetLoader:
    """Handles loading class names from different dataset formats."""
    
    @staticmethod
    def load_classnames(dataset_name: str, data_dir: str) -> Tuple[List[str], str]:
        """
        Load class names for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            data_dir: Root directory containing datasets
            
        Returns:
            Tuple of (list of class names, object type description)
        """
        dataset_path = Path(data_dir) / dataset_name
        
        if dataset_name == "imagenet":
            return DatasetLoader._load_imagenet(dataset_path)
        elif dataset_name == "caltech-101":
            return DatasetLoader._load_caltech101(dataset_path)
        elif dataset_name == "oxford_pets":
            return DatasetLoader._load_oxford_pets(dataset_path)
        elif dataset_name == "stanford_cars":
            return DatasetLoader._load_stanford_cars(dataset_path)
        elif dataset_name == "oxford_flowers":
            return DatasetLoader._load_oxford_flowers(dataset_path)
        elif dataset_name == "food-101":
            return DatasetLoader._load_food101(dataset_path)
        elif dataset_name == "fgvc_aircraft":
            return DatasetLoader._load_fgvc_aircraft(dataset_path)
        elif dataset_name == "sun397":
            return DatasetLoader._load_sun397(dataset_path)
        elif dataset_name == "dtd":
            return DatasetLoader._load_dtd(dataset_path)
        elif dataset_name == "eurosat":
            return DatasetLoader._load_eurosat(dataset_path)
        elif dataset_name == "ucf101":
            return DatasetLoader._load_ucf101(dataset_path)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
    
    @staticmethod
    def _load_imagenet(dataset_path: Path) -> Tuple[List[str], str]:
        """Load ImageNet class names."""
        classnames_path = dataset_path / "classnames.txt"
        with open(classnames_path, 'r') as f:
            classnames = [line.strip() for line in f.readlines()]
        return classnames, "object"
    
    @staticmethod
    def _load_caltech101(dataset_path: Path) -> Tuple[List[str], str]:
        """Load Caltech101 class names."""
        split_file = dataset_path / "split_zhou_Caltech101.json"
        with open(split_file, 'r') as f:
            data = json.load(f)
        
        # The format is different - train contains a list of lists: [filepath, class_id, class_name]
        # Extract the class names (3rd element in each inner list)
        classnames = []
        for item in data['train']:
            if len(item) >= 3:  # Ensure there are 3 elements
                class_name = item[2]
                if class_name not in classnames:
                    classnames.append(class_name)
        
        return sorted(classnames), "object"
    
    @staticmethod
    def _load_oxford_pets(dataset_path: Path) -> Tuple[List[str], str]:
        """Load Oxford Pets class names."""
        split_file = dataset_path / "split_zhou_OxfordPets.json"
        with open(split_file, 'r') as f:
            data = json.load(f)
        # Extract the third element (index 2) from each item in the train list
        classnames = sorted(list(set([item[2] for item in data['train']])))
        return classnames, "pet"

    @staticmethod
    def _load_stanford_cars(dataset_path: Path) -> Tuple[List[str], str]:
        """Load Stanford Cars class names."""
        split_file = dataset_path / "split_zhou_StanfordCars.json"
        with open(split_file, 'r') as f:
            data = json.load(f)
        # Extract the third element (index 2) from each item in the train list
        classnames = sorted(list(set([item[2] for item in data['train']])))
        return classnames, "car"
    
    @staticmethod
    def _load_oxford_flowers(dataset_path: Path) -> Tuple[List[str], str]:
        """Load Oxford Flowers class names."""
        cat_to_name_file = dataset_path / "cat_to_name.json"
        with open(cat_to_name_file, 'r') as f:
            cat_to_name = json.load(f)
        # Convert from {'1': 'pink primrose', '2': 'hard-leaved pocket orchid'} format
        classnames = [name.strip() for name in cat_to_name.values()]
        return classnames, "flower"
    
    @staticmethod
    def _load_food101(dataset_path: Path) -> Tuple[List[str], str]:
        """Load Food101 class names."""
        split_file = dataset_path / "split_zhou_Food101.json"
        with open(split_file, 'r') as f:
            data = json.load(f)
        # Extract the third element (index 2) from each item in the train list
        classnames = sorted(list(set([item[2] for item in data['train']])))
        # Clean class names by replacing underscores with spaces
        classnames = [name.replace('_', ' ') for name in classnames]
        return classnames, "food"
    
    @staticmethod
    def _load_fgvc_aircraft(dataset_path: Path) -> Tuple[List[str], str]:
        """Load FGVC Aircraft class names."""
        variant_file = dataset_path / "variants.txt"
        with open(variant_file, 'r') as f:
            classnames = [line.strip() for line in f.readlines()]
        return classnames, "aircraft"
    
    @staticmethod
    def _load_sun397(dataset_path: Path) -> Tuple[List[str], str]:
        """Load SUN397 class names."""
        split_file = dataset_path / "split_zhou_SUN397.json"
        with open(split_file, 'r') as f:
            data = json.load(f)
        classnames = sorted(list(set([item['classname'] for item in data['train']])))
        return classnames, "scene"
    
    @staticmethod
    def _load_dtd(dataset_path: Path) -> Tuple[List[str], str]:
        """Load DTD (Describable Textures) class names."""
        split_file = dataset_path / "split_zhou_DescribableTextures.json"
        with open(split_file, 'r') as f:
            data = json.load(f)
        # Extract the third element (index 2) from each item in the train list
        classnames = sorted(list(set([item[2] for item in data['train']])))
        return classnames, "texture"
    
    @staticmethod
    def _load_eurosat(dataset_path: Path) -> Tuple[List[str], str]:
        """Load EuroSAT class names."""
        split_file = dataset_path / "split_zhou_EuroSAT.json"
        with open(split_file, 'r') as f:
            data = json.load(f)
        # Extract the third element (index 2) from each item in the train list
        classnames = sorted(list(set([item[2] for item in data['train']])))
        return classnames, "satellite imagery"
    
    
    @staticmethod
    def _load_ucf101(dataset_path: Path) -> Tuple[List[str], str]:
        """Load UCF101 class names."""
        split_file = dataset_path / "split_zhou_UCF101.json"
        with open(split_file, 'r') as f:
            data = json.load(f)
        # Extract the third element (index 2) from each item in the train list
        classnames = sorted(list(set([item[2] for item in data['train']])))
        return classnames, "action"

def main(args):
    # Initialize attribute generator
    generator = AttributeGenerator(
        api_key=args.openai_key,
        save_dir=args.output_dir
    )
    
    # Process each dataset in the list
    for dataset_name in args.datasets:
        try:
            print(f"\n=== Processing dataset: {dataset_name} ===")
            
            # Check if attributes already exist
            if not args.force and generator.load_dataset_attributes(dataset_name):
                print(f"Attributes for {dataset_name} already exist. Use --force to regenerate.")
                continue
            
            # Load class names
            classnames, type_name = DatasetLoader.load_classnames(
                dataset_name=dataset_name,
                data_dir=args.data_dir
            )
            
            print(f"Loaded {len(classnames)} classes for {dataset_name}")
            print(f"Object type: {type_name}")
            
            # Generate attributes
            print(f"Generating attributes...")
            attributes = generator.generate_and_save_dataset_attributes(
                classnames=classnames,
                type_name=type_name,
                dataset_name=dataset_name
            )
            
            print(f"Successfully generated attributes for {len(attributes)} classes in {dataset_name}")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    print(f"\nAll attribute generation completed. Results saved in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate attributes for multiple datasets using GPT")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data",
        help="Root directory containing all datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="attributes",
        help="Directory to save generated attributes"
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        required=True,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["oxford_flowers"],
        choices=[
            "imagenet", "caltech-101", "oxford_pets", "stanford_cars", 
            "oxford_flowers", "food-101", "fgvc_aircraft", "sun397", 
            "dtd", "eurosat", "ucf101"
        ],
        help="List of datasets to process"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of attributes even if they already exist"
    )
    
    args = parser.parse_args()
    main(args)