import torch
from pathlib import Path
from dataset import FontDataset
from torch.utils.data import DataLoader
from train import FontSimilarityModel
from typing import List, Dict, Tuple

def find_similar_fonts(self, query_idx: int, centroids: Dict[int, torch.Tensor], 
                          k: int = 5) -> List[Tuple[int, float]]:
        query_centroid = centroids[query_idx]
        similarities = []
        
        for idx, centroid in centroids.items():
            if idx != query_idx:
                sim = torch.nn.functional.cosine_similarity(query_centroid.unsqueeze(0), 
                                        centroid.unsqueeze(0))
                similarities.append((idx, sim.item()))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

# Example usage
def main():
    # Sample text to render
    text_samples = [
        "The quick brown fox jumps over the lazy dog",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz",
        "0123456789",
    ]
    
    # Get list of font files
    font_dir = Path("fonts")  # Update with your font directory
    font_paths = list(font_dir.glob("*.ttf"))
    
    # TODO pass in fonts.txt here as a parameter, ideally add arg to the fonts file and
    # make the default fonts.txt
    # Create dataset and dataloader
    dataset = FontDataset()#font_paths, text_samples)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize and train model
    model = FontSimilarityModel()
    #model = torch.compile(model)
    breakpoint()
    num_epochs = 10
    
    for epoch in range(num_epochs):
        loss = model.train_epoch(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Compute font centroids
    centroids = model.compute_centroids(dataloader)
    
    # Find similar fonts for a query font
    query_idx = 0  # Example query font index
    similar_fonts = model.find_similar_fonts(query_idx, centroids)
    
    print(f"\nMost similar fonts to {font_paths[query_idx].name}:")
    for idx, similarity in similar_fonts:
        print(f"{font_paths[idx].name}: {similarity:.4f}")

if __name__ == "__main__":
    main()