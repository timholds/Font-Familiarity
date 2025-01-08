from flask import Flask, request, jsonify, render_template
import os
import torch
import numpy as np
from PIL import Image
import io
import torchvision.transforms as transforms
from ml.model import SimpleCNN
import torch.nn.functional as F
import traceback
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
class_embeddings = None
device = None
label_mapping = None
is_initialized = False

def load_model_and_embeddings(model_path: str, 
                            embeddings_path: str, 
                            label_mapping_path: str) -> None:
    """Initialize model and load pre-computed class embeddings."""
    global model, class_embeddings, device, label_mapping, is_initialized
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Verify files exist
        for path in [model_path, embeddings_path, label_mapping_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
        
        # Load model state
        logger.info(f"Loading model from {model_path}")
        state = torch.load(model_path, map_location=device)
        state_dict = state['model_state_dict']
        
        # Extract architecture parameters
        embedding_weight = state_dict['embedding_layer.0.weight']
        classifier_weight = state_dict['classifier.weight']
        
        embedding_dim = embedding_weight.shape[0]
        flatten_dim = embedding_weight.shape[1]
        num_classes = classifier_weight.shape[0]
        
        logger.info(f"Model architecture: embedding_dim={embedding_dim}, "
                   f"flatten_dim={flatten_dim}, num_classes={num_classes}")
        
        # Initialize and load model
        model = SimpleCNN(
            num_classes=num_classes,
            embedding_dim=embedding_dim
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Load embeddings
        logger.info(f"Loading embeddings from {embeddings_path}")
        class_embeddings = torch.from_numpy(np.load(embeddings_path)).to(device)
        
        # Load label mapping
        logger.info(f"Loading label mapping from {label_mapping_path}")
        label_mapping = np.load(label_mapping_path, allow_pickle=True).item()
        
        is_initialized = True
        logger.info("Model initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Convert uploaded image bytes to tensor."""
    try:
        # Open image and convert to grayscale
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        logger.info(f"Loaded image, size: {image.size}")
        
        # Define preprocessing
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        # Apply preprocessing and add batch dimension
        tensor = transform(image).unsqueeze(0)
        logger.info(f"Preprocessed tensor shape: {tensor.shape}")
        
        return tensor.to(device)
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

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
    logger.info("Received request for index page")
    logger.info(f"Current working directory: {os.path.abspath(os.curdir)}")
    logger.info(f"Template folder: {os.path.abspath(os.path.join(os.curdir, 'templates'))}")
    
    try:
        templates_dir = os.path.join(os.curdir, 'templates')
        if os.path.exists(templates_dir):
            logger.info(f"Available templates: {os.listdir(templates_dir)}")
        else:
            logger.error(f"Templates directory not found: {templates_dir}")
            return "Error: Templates directory not found", 500
            
        return render_template('frontend.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for font prediction using both similarity and classification approaches."""
    if not is_initialized:
        return jsonify({'error': 'Model not initialized'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get and preprocess image
        image_bytes = request.files['image'].read()
        logger.info(f"Received image, size: {len(image_bytes)} bytes")
        
        image_tensor = preprocess_image(image_bytes)
        
        with torch.no_grad():
            # Get embedding and classifier output
            embedding = model.get_embedding(image_tensor)
            logger.info(f"Generated embedding, shape: {embedding.shape}")
            
            logits = model.classifier(embedding)
            logger.info(f"Generated logits, shape: {logits.shape}")
            
            # Get predictions using both methods
            emb_indices, emb_scores = get_top_k_similar_fonts(embedding)
            cls_indices, cls_probs = get_top_k_predictions(logits)
            
            logger.info(f"Embedding indices: {emb_indices}")
            logger.info(f"Classifier indices: {cls_indices}")
            
            # Validate indices before conversion
            all_indices = np.concatenate([emb_indices, cls_indices])
            invalid_indices = [idx for idx in all_indices if idx not in label_mapping]
            if invalid_indices:
                raise ValueError(f"Found invalid indices not in label mapping: {invalid_indices}")
            
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
        logger.error(f"Error processing prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f"Error processing image: {str(e)}",
            'details': traceback.format_exc()
        }), 500
    
def main():
    """Initialize and run the Flask application."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained model .pt file")
    parser.add_argument("--embeddings_path", default="class_embeddings.npy",
                       help="Path to class embeddings .npy file")
    parser.add_argument("--label_mapping_path", default="font_dataset_npz/label_mapping.npy",
                       help="Path to label_mapping.npy file")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    logger.info("\nInitializing Flask app...")
    logger.info(f"Port: {args.port}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Embeddings path: {args.embeddings_path}")
    logger.info(f"Label mapping path: {args.label_mapping_path}")
    
    # Verify required directories exist
    for directory in ['templates', 'static']:
        dir_path = os.path.join(os.curdir, directory)
        if not os.path.exists(dir_path):
            logger.error(f"Required directory not found: {dir_path}")
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    # Load model and embeddings
    try:
        load_model_and_embeddings(
            args.model_path,
            args.embeddings_path,
            args.label_mapping_path
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    logger.info(f"\nStarting Flask server on port {args.port}...")
    logger.info("You can access the app at:")
    logger.info(f"  http://localhost:{args.port}")
    logger.info(f"  http://127.0.0.1:{args.port}")
    logger.info("Use Ctrl+C to stop the server\n")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=args.port, debug=True)

if __name__ == '__main__':
    main()