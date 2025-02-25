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
    # Define short templates that prompt for 5 attributes each, 
    # ensuring each attribute is kept under 50 tokens.
    templates = [
        # Template 1
        (
            "Q: Provide 5 short visual attributes of a {class} ({type}), each <50 tokens.\n"
            "A:"
        ),
        # Template 2
        (
            "Q: Give 5 brief traits that differentiate a {class} from similar {type}s, each <50 tokens.."
            "A:"
        ),
        # Template 3
        (
            "Q: Describe 5 distinct indicators that help identify a {class} ({type}), each <50 tokens.\n"
            "A:"
        )
    ]
    
    # Initialize attribute generator with the custom templates.
    # Make sure your AttributeGenerator can handle these extra parameters.
    generator = AttributeGenerator(
        api_key=args.openai_key,
        save_dir=os.path.join(args.data_dir, "attributes"),
        templates=templates,     # Pass in the list of templates
        max_tokens=args.max_tokens  # Ensure each attribute is kept short
    )
    
    # Read class names from dataset
    classnames = read_flower_classes(
        os.path.join(args.data_dir, "cat_to_name.json")
    )
    
    print(f"Generating attributes for {len(classnames)} flower classes...")
    
    # Generate attributes for all classes
    # attributes_per_template=5 indicates we want 5 attributes from each template
    attributes = generator.generate_and_save_dataset_attributes(
        classnames=classnames,
        type_name="flower",
        dataset_name="oxford_flowers",
        attributes_per_template=5
    )
    
    print(f"Successfully generated and saved attributes for {len(attributes)} classes")
    print(f"Attributes saved in {generator.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--openai_key",
        type=str,
        required=True,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens for each attribute (to keep them concise)"
    )
    
    args = parser.parse_args()
    main(args)
