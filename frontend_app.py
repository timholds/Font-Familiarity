from flask import Flask, request, jsonify, render_template
import os
import torch
import numpy as np
from PIL import Image
import io
import torchvision.transforms as transforms
from ml.model import SimpleCNN
import torch.nn.functional as F

app = Flask(__name__)

"""
Model architecture flow:
1. Input image (1, 64, 64)
2. CNN layers -> (128, 4, 4)
3. Flatten -> 2048
4. Embedding layer -> 128
5. Classifier -> num_classes

We use the 128-dimensional embeddings for similarity comparison.
"""

# Global variables
model = None
class_embeddings = None  # Shape: [num_classes, embedding_dim]
device = None
label_mapping = None

def load_model_and_embeddings(model_path: str, 
                            embeddings_path: str, 
                            label_mapping_path: str) -> None:
    """
    Initialize model and load pre-computed class embeddings.
    """
    global model, class_embeddings, device, label_mapping
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model state
    state = torch.load(model_path, map_location=device)
    state_dict = state['model_state_dict']
    
    # Extract architecture parameters from state dict
    embedding_weight = state_dict['embedding_layer.0.weight']  # Shape: [embedding_dim, flatten_dim]
    classifier_weight = state_dict['classifier.weight']        # Shape: [num_classes, embedding_dim]
    
    # Get dimensions from the weights
    embedding_dim = embedding_weight.shape[0]  # 128 (dimension of learned features)
    flatten_dim = embedding_weight.shape[1]    # 2048 (128 channels * 4 * 4)
    num_classes = classifier_weight.shape[0]   # Number of font classes
    
    print("\nModel architecture from checkpoint:")
    print(f"- Flattened CNN features: {flatten_dim} (128 channels * 4 * 4 spatial)")
    print(f"- Embedding dimension: {embedding_dim}")
    print(f"- Number of classes: {num_classes}")
    
    # Initialize model with correct parameters
    model = SimpleCNN(
        num_classes=num_classes,
        embedding_dim=embedding_dim
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load pre-computed class embeddings
    print(f"\nLoading embeddings from: {embeddings_path}")
    class_embeddings = torch.from_numpy(np.load(embeddings_path)).to(device)
    
    # Load label mapping
    print(f"Loading label mapping from: {label_mapping_path}")
    label_mapping = np.load(label_mapping_path, allow_pickle=True).item()
    
    print("\nModel and embeddings loaded successfully!")
    print(f"Number of classes: {len(label_mapping)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Class embeddings shape: {class_embeddings.shape}")

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Convert uploaded image bytes to tensor."""
    # Open image and convert to grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Define preprocessing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    # Apply preprocessing and add batch dimension
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor.to(device)

def get_top_k_similar_fonts(query_embedding: torch.Tensor, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Find k most similar fonts using embedding similarity."""
    # Normalize query embedding for cosine similarity
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    # Compute cosine similarity with all class embeddings
    similarities = torch.mm(query_embedding, class_embeddings.t())
    
    # Get top k similarities and indices
    top_k_similarities, top_k_indices = similarities[0].topk(k)
    
    return top_k_indices.cpu().numpy(), top_k_similarities.cpu().numpy()

def get_top_k_predictions(logits: torch.Tensor, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Get top k predictions from classifier."""
    probabilities = F.softmax(logits, dim=1)
    top_k_probs, top_k_indices = probabilities[0].topk(k)
    
    return top_k_indices.cpu().numpy(), top_k_probs.cpu().numpy()

@app.route('/test')
def test():
    """Test route to verify Flask is working."""
    return "Flask server is running!"

@app.route('/')
def index():
    """Serve the main page."""
    print("\n=== Request received for index page ===")
    print("Current working directory:", os.path.abspath(os.curdir))
    print("Template folder:", os.path.abspath(os.path.join(os.curdir, 'templates')))
    print("Available templates:", os.listdir('templates'))
    
    try:
        print("Attempting to serve frontend.html template...")
        return render_template('frontend.html')
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return f"Error: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for font prediction using both similarity and classification approaches."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get and preprocess image
        image_bytes = request.files['image'].read()
        image_tensor = preprocess_image(image_bytes)
        
        with torch.no_grad():
            # Get embedding (128-dim) and classifier output
            embedding = model.get_embedding(image_tensor)  # Shape: [1, embedding_dim]
            logits = model.classifier(embedding)           # Shape: [1, num_classes]
            
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
        print(f"Error processing prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained model .pt file")
    parser.add_argument("--embeddings_path", default="class_embeddings.npy",
                       help="Path to class embeddings .npy file")
    parser.add_argument("--label_mapping_path", default="font_dataset_npz/label_mapping.npy",
                       help="Path to label_mapping.npy file")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    print("\nInitializing Flask app...")
    print(f"Port: {args.port}")
    print(f"Model path: {args.model_path}")
    print(f"Embeddings path: {args.embeddings_path}")
    print(f"Label mapping path: {args.label_mapping_path}")
    
    # Load model and embeddings
    load_model_and_embeddings(
        args.model_path,
        args.embeddings_path,
        args.label_mapping_path
    )
    
    print(f"\nStarting Flask server on port {args.port}...")
    print(f"You can access the app at:")
    print(f"  http://localhost:{args.port}")
    print(f"  http://127.0.0.1:{args.port}")
    print("Use Ctrl+C to stop the server\n")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=args.port, debug=True)