# inference.py
import torch
import torch.nn.functional as F
from PIL import Image
from model import FontEmbeddingModel  # your model class

class FontMatcher:
    def __init__(self, model_path, class_embeddings_path, device='cuda'):
        self.device = device
        self.model = FontEmbeddingModel(num_classes=700).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.class_embeddings = torch.load(class_embeddings_path).to(device)
        
    def find_closest_fonts(self, query_image, k=5):
        """Find k most similar fonts to the query image"""
        with torch.no_grad():
            # Get query embedding
            query_embedding = self.model.get_embedding(query_image.unsqueeze(0))
            
            # Compute cosine similarities
            similarities = F.cosine_similarity(
                query_embedding.unsqueeze(1),
                self.class_embeddings.unsqueeze(0),
                dim=2
            )
            
            # Get top k most similar
            top_k_similarities, top_k_indices = similarities[0].topk(k)
            
            return top_k_indices, top_k_similarities

    def preprocess_image(self, image_path):
        """Preprocess a single image for inference"""
        # Add your image preprocessing code here
        pass

# Usage example
if __name__ == "__main__":
    matcher = FontMatcher(
        model_path='model_weights.pt',
        class_embeddings_path='class_embeddings.pt'
    )
    
    # Example usage
    image = matcher.preprocess_image('test_font.png')
    indices, similarities = matcher.find_closest_fonts(image, k=5)