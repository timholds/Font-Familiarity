from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
import io
import torchvision.transforms as transforms
from model import SimpleCNN
import torch.nn.functional as F

app = Flask(__name__)

# Global variables to store model and embeddings
model = None
class_embeddings = None
device = None
label_mapping = None

def load_model_and_embeddings(model_path, embeddings_path, label_mapping_path):
    """Initialize model and load embeddings."""
    global model, class_embeddings, device, label_mapping
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    state = torch.load(model_path, map_location=device)
    model = SimpleCNN(
        num_classes=state['num_classes'],
        embedding_dim=state['model_state_dict']['embedding_layer.0.weight'].shape[1]
    ).to(device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    # Load pre-computed class embeddings
    class_embeddings = torch.from_numpy(np.load(embeddings_path)).to(device)
    
    # Load label mapping
    label_mapping = np.load(label_mapping_path, allow_pickle=True).item()
    
    print("Model and embeddings loaded successfully!")
    print(f"Number of classes: {len(label_mapping)}")
    print(f"Embedding dimension: {class_embeddings.shape[1]}")

def preprocess_image(image_bytes):
    """Convert uploaded image bytes to tensor."""
    # Open image and convert to grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Define preprocessing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    # Apply preprocessing
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor.to(device)

def get_top_k_similar_fonts(query_embedding, k=5):
    """Find k most similar fonts using embedding similarity."""
    # Normalize query embedding
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    # Compute cosine similarity with all class embeddings
    similarities = torch.mm(query_embedding, class_embeddings.t())
    
    # Get top k similarities and indices
    top_k_similarities, top_k_indices = similarities[0].topk(k)
    
    return top_k_indices.cpu().numpy(), top_k_similarities.cpu().numpy()

def get_top_k_predictions(logits, k=5):
    """Get top k predictions from classifier."""
    probabilities = F.softmax(logits, dim=1)
    top_k_probs, top_k_indices = probabilities[0].topk(k)
    
    return top_k_indices.cpu().numpy(), top_k_probs.cpu().numpy()

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for font prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get image from request
        image_bytes = request.files['image'].read()
        image_tensor = preprocess_image(image_bytes)
        
        with torch.no_grad():
            # Get embedding and classifier output
            embedding = model.get_embedding(image_tensor)
            logits = model.classifier(embedding)
            
            # Get predictions using both methods
            emb_indices, emb_scores = get_top_k_similar_fonts(embedding)
            cls_indices, cls_probs = get_top_k_predictions(logits)
            
            # Convert indices to font names
            embedding_results = [
                {
                    'font': label_mapping[int(idx)],
                    'similarity': float(score)
                } for idx, score in zip(emb_indices, emb_scores)
            ]
            
            classifier_results = [
                {
                    'font': label_mapping[int(idx)],
                    'probability': float(prob)
                } for idx, prob in zip(cls_indices, cls_probs)
            ]
            
            return jsonify({
                'embedding_similarity': embedding_results,
                'classifier_predictions': classifier_results
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained model .pt file")
    parser.add_argument("--embeddings_path", required=True, help="Path to class embeddings .npy file")
    parser.add_argument("--label_mapping_path", required=True, help="Path to label_mapping.npy file")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    
    # Load model and embeddings
    load_model_and_embeddings(
        args.model_path,
        args.embeddings_path,
        args.label_mapping_path
    )
    
    # Start Flask app
    app.run(host='0.0.0.0', port=args.port)