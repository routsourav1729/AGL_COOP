import os
import json
import openai
from typing import List, Dict, Optional
import torch
import clip
from pathlib import Path

class AttributeGenerator:
    """Attribute generator for vision-language models following ArGue paper methodology."""
    
    def __init__(self, api_key: str, save_dir: str = "attributes"):
        """
        Initialize the attribute generator.
        
        Args:
            api_key: OpenAI API key
            save_dir: Directory to save generated attributes
        """
        self.api_key = api_key
        openai.api_key = api_key
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Templates from ArGue paper
        self.templates = [
            {
                "system": "You are an expert at describing visual attributes of objects in images.",
                "user": "Describe what a {type} {class_name} looks like in a photo, list {num} pieces?"
            },
            {
                "system": "You are an expert at describing visual attributes of objects in images.",
                "user": "Visually describe a {class_name}, a type of {type}, list {num} pieces?"
            },
            {
                "system": "You are an expert at describing visual attributes of objects in images.", 
                "user": "How to distinguish a {class_name} which is a {type}, list {num} pieces?"
            }
        ]
        
        # Example prompts for in-context learning
        self.examples = [
            {
                "user": "Describe what an animal giraffe looks like in a photo, list 6 pieces?",
                "assistant": "There are 6 useful visual features for a giraffe in a photo:\n- covered with a spotted coat\n- has a short, stocky body\n- has a long neck\n- owns a small neck to its body\n- is yellow or brown in color\n- have a black tufted tail"
            }
        ]

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
        dataset_name: str
    ) -> Dict[str, List[str]]:
        """
        Generate and save attributes for all classes in a dataset.
        
        Args:
            classnames: List of class names
            type_name: Type of object (e.g., "flower")
            dataset_name: Name of the dataset for saving
            
        Returns:
            Dictionary mapping class names to attributes
        """
        all_class_attributes = {}
        
        for class_name in classnames:
            # Generate attributes for current class
            attributes = self.generate_class_attributes(
                class_name=class_name,
                type_name=type_name
            )
            all_class_attributes[class_name] = attributes
            
        # Save results
        save_path = self.save_dir / f"{dataset_name}_attributes.json"
        with open(save_path, 'w') as f:
            json.dump(all_class_attributes, f, indent=2)
            
        return all_class_attributes

    def load_dataset_attributes(self, dataset_name: str) -> Optional[Dict[str, List[str]]]:
        """Load previously generated attributes for a dataset."""
        path = self.save_dir / f"{dataset_name}_attributes.json"
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None