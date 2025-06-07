import os
import argparse
import json
from utils.attribute_gen import AttributeGenerator

def read_flower_classes(json_path):
    """Read class names from Oxford Flowers dataset."""
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    # Convert from {'1': 'pink primrose', '2': 'hard-leaved pocket orchid'} format
    classnames = [name.strip() for name in cat_to_name.values()]
    return classnames

def main(args):
    # Initialize attribute generator
    generator = AttributeGenerator(
        api_key=args.openai_key,
        save_dir=os.path.join(args.data_dir, "attributes")
    )
    
    # Read class names from dataset
    classnames = read_flower_classes(
        os.path.join(args.data_dir, "cat_to_name.json")
    )
    
    print(f"Generating attributes for {len(classnames)} flower classes...")
    
    # Generate attributes for all classes
    attributes = generator.generate_and_save_dataset_attributes(
        classnames=classnames,
        type_name="flower",
        dataset_name="oxford_flowers"
    )
    
    print(f"Successfully generated and saved attributes for {len(attributes)} classes")
    print(f"Attributes saved in {generator.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data",
        help="root directory containing datasets"
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        required=True,
        help="OpenAI API key"
    )
    
    args = parser.parse_args()
    main(args)