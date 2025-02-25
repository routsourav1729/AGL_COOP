class OxfordFlowersProcessor(AttributeGenerator, AttributeSampler):
    """
    Oxford Flowers processor that inherits both AttributeGenerator and AttributeSampler.
    This class is designed to work exclusively with the Oxford Flowers dataset.
    """
    def __init__(self, api_key: str, data_dir: str, split_file: str, image_root: str, num_clusters: int = 3, max_tokens: int = 50, device: str = None):
        self.data_dir = data_dir
        # Define file paths specific to Oxford Flowers
        attr_save_dir = os.path.join(data_dir, "attributes")
        self.cat_to_name_file = os.path.join(data_dir, "cat_to_name.json")
        self.generated_attr_file = os.path.join(attr_save_dir, "oxford_flowers_attributes.json")
        self.sampled_attr_file = os.path.join(attr_save_dir, "oxford_flowers_sampled_attributes.json")
        # Initialize both parent classes explicitly
        AttributeGenerator.__init__(self, api_key=api_key, save_dir=attr_save_dir, max_tokens=max_tokens)
        AttributeSampler.__init__(self, split_file=split_file, attr_file=self.generated_attr_file,
                                    image_root=image_root, output_file=self.sampled_attr_file,
                                    num_clusters=num_clusters, device=device)

    def read_flower_classes(self) -> list:
        """Reads Oxford Flowers class names from cat_to_name.json."""
        with open(self.cat_to_name_file, 'r') as f:
            cat_to_name = json.load(f)
        return [name.strip() for name in cat_to_name.values()]

    def process(self):
        """Generate attributes then sample discriminative attributes for Oxford Flowers."""
        classes = self.read_flower_classes()
        print(f"Generating attributes for {len(classes)} Oxford Flowers classes...")
        generated = self.generate_and_save_dataset_attributes(classnames=classes, type_name="flower", dataset_name="oxford_flowers")
        print(f"Generated attributes saved at {self.generated_attr_file}")
        print("Sampling attributes using CLIP embeddings...")
        sampled = self.sample_attributes()
        print(f"Sampled attributes saved at {self.sampled_attr_file}")
        return generated, sampled

# -------------------------------------------------------------------
# Main for Oxford Flowers
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root directory containing datasets")
    parser.add_argument("--openai_key", type=str, required=True,
                        help="OpenAI API key")
    parser.add_argument("--split_file", type=str,
                        default="/home/sourav/ALL_FILES/Thesis/attribute-guided-learning/data/oxford_flowers/split_zhou_OxfordFlowers.json")
    parser.add_argument("--image_root", type=str,
                        default="/home/sourav/ALL_FILES/Thesis/attribute-guided-learning/data/oxford_flowers/jpg")
    parser.add_argument("--num_clusters", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=50)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    processor = OxfordFlowersProcessor(
        api_key=args.openai_key,
        data_dir=args.data_dir,
        split_file=args.split_file,
        image_root=args.image_root,
        num_clusters=args.num_clusters,
        max_tokens=args.max_tokens
    )
    processor.process()
