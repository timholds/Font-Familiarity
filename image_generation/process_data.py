import os
from pathlib import Path
from PIL import Image
import datasets
from datasets import Dataset, Image as DsImage
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
from dataclasses import dataclass
import torch
from torch.nn import functional as F

@dataclass
class FontMetadata:
    """Metadata for a font including retrieval information."""
    font_name: str
    font_path: str
    style_attributes: Optional[Dict] = None
    embedding: Optional[np.ndarray] = None

class FontDatasetManager:
    """Manages font dataset for both training and retrieval."""
    
    def __init__(
        self,
        root_dir: str,
        output_dir: str = "processed_dataset",
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.font_metadata: Dict[str, FontMetadata] = {}
        self.num_classes: Optional[int] = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def prepare_dataset(self) -> Dataset:
        """Prepare the dataset for training and retrieval."""
        image_paths = []
        labels = []
        metadata_list = []
        
        # Collect all font directories
        font_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.num_classes = len(font_dirs)
        
        # Create font name to index mapping
        self.font_to_idx = {font_dir.name: idx for idx, font_dir in enumerate(font_dirs)}
        self.idx_to_font = {idx: name for name, idx in self.font_to_idx.items()}
        
        # Process each font directory
        for font_dir in font_dirs:
            font_name = font_dir.name
            
            # Initialize font metadata
            self.font_metadata[font_name] = FontMetadata(
                font_name=font_name,
                font_path=str(font_dir)
            )
            
            # Process each image
            for img_path in font_dir.glob("*.png"):
                image_paths.append(str(img_path))
                labels.append(self.font_to_idx[font_name])
                
                metadata_list.append({
                    "font_name": font_name,
                    "original_path": str(img_path),
                    "file_name": img_path.name
                })
        
        # Save mappings and metadata
        self._save_metadata()
        
        def process_example(example: Dict) -> Dict:
            """Process a single example, including one-hot encoding."""
            # Load and resize image
            image = Image.open(example["image_path"]).convert("RGB")
            image = image.resize(self.image_size)
            
            # Create one-hot encoded label
            label_onehot = np.zeros(self.num_classes)
            label_onehot[example["label"]] = 1
            
            return {
                "image": image,
                "label": example["label"],
                "label_onehot": label_onehot,
                "font_name": example["metadata"]["font_name"],
                "metadata": example["metadata"]
            }
        
        # Create and process dataset
        dataset = Dataset.from_dict({
            "image_path": image_paths,
            "label": labels,
            "metadata": metadata_list
        })
        
        dataset = dataset.map(
            process_example,
            remove_columns=dataset.column_names,
            num_proc=4
        )
        
        # Save processed dataset
        dataset.save_to_disk(self.output_dir)
        return dataset
    
    def _save_metadata(self):
        """Save font mappings and metadata."""
        metadata_path = self.output_dir / "font_metadata.json"
        
        metadata_dict = {
            "font_to_idx": self.font_to_idx,
            "idx_to_font": self.idx_to_font,
            "num_classes": self.num_classes,
            "font_metadata": {
                name: {
                    "font_name": meta.font_name,
                    "font_path": meta.font_path,
                    "style_attributes": meta.style_attributes
                }
                for name, meta in self.font_metadata.items()
            }
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)
    
    @classmethod
    def load(cls, dataset_dir: str) -> "FontDatasetManager":
        """Load a previously prepared dataset manager."""
        manager = cls(root_dir="", output_dir=dataset_dir)
        
        # Load metadata
        metadata_path = Path(dataset_dir) / "font_metadata.json"
        with open(metadata_path) as f:
            metadata_dict = json.load(f)
        
        manager.font_to_idx = metadata_dict["font_to_idx"]
        manager.idx_to_font = metadata_dict["idx_to_font"]
        manager.num_classes = metadata_dict["num_classes"]
        
        # Reconstruct font metadata
        for name, meta in metadata_dict["font_metadata"].items():
            manager.font_metadata[name] = FontMetadata(**meta)
        
        return manager
    
    def get_font_name(self, idx: int) -> str:
        """Get font name from index."""
        return self.idx_to_font[idx]
    
    def get_font_idx(self, font_name: str) -> int:
        """Get index from font name."""
        return self.font_to_idx[font_name]
    
    def store_embeddings(self, embeddings: np.ndarray, font_names: List[str]):
        """Store embeddings for fonts for later similarity lookup."""
        for emb, name in zip(embeddings, font_names):
            if name in self.font_metadata:
                self.font_metadata[name].embedding = emb
    
    def find_similar_fonts(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        metric: str = "cosine"
    ) -> List[Tuple[str, float]]:
        """Find k most similar fonts to a query embedding."""
        # Collect all stored embeddings
        font_names = []
        embeddings = []
        
        for name, meta in self.font_metadata.items():
            if meta.embedding is not None:
                font_names.append(name)
                embeddings.append(meta.embedding)
        
        if not embeddings:
            raise ValueError("No embeddings stored. Call store_embeddings first.")
        
        embeddings = np.stack(embeddings)
        
        # Calculate similarities
        if metric == "cosine":
            similarities = F.cosine_similarity(
                torch.tensor(query_embedding)[None, :],
                torch.tensor(embeddings)
            )[0].numpy()
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
        
        # Get top k similar fonts
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        return [(font_names[idx], similarities[idx]) for idx in top_k_idx]

# Example usage:
if __name__ == "__main__":
    # Prepare dataset
    manager = FontDatasetManager("font-images")
    dataset = manager.prepare_dataset()
    
    # Split dataset
    splits = dataset.train_test_split(
        train_size=0.8,
        test_size=0.2,
        seed=42
    )
    
    print(f"\nPrepared dataset with {manager.num_classes} font classes")
    print(f"Train split: {len(splits['train'])} examples")
    print(f"Test split: {len(splits['test'])} examples")
    
    # Example of storing and retrieving similar fonts
    # (This would normally be done after training)
    dummy_embeddings = np.random.randn(manager.num_classes, 128)
    font_names = list(manager.font_metadata.keys())
    manager.store_embeddings(dummy_embeddings, font_names)
    
    # Find similar fonts
    query_embedding = np.random.randn(128)
    similar_fonts = manager.find_similar_fonts(query_embedding, k=5)
    print("\nExample similar fonts:")
    for font_name, similarity in similar_fonts:
        print(f"{font_name}: {similarity:.3f}")