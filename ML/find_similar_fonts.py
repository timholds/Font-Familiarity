import torch

def compute_font_embedding(model, font_images):
    """
    Compute average latent representation for a font
    """
    with torch.no_grad():
        embeddings = model(font_images)
        return torch.mean(embeddings, dim=0)

def find_similar_fonts(query_embedding, font_embeddings, k=5):
    """
    Find k most similar fonts using cosine similarity
    """
    similarities = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0),
        font_embeddings
    )
    top_k = torch.topk(similarities, k)
    return top_k.indices

# Feature based similarity
def get_class_centroids(model, dataloader):
    centroids = {}  # class_id -> average embedding
    class_samples = {}  # class_id -> count
    
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            embeddings = model.encode(images)  # shape: (batch_size, embedding_dim)
            
            for emb, label in zip(embeddings, labels):
                if label.item() not in centroids:
                    centroids[label.item()] = emb
                    class_samples[label.item()] = 1
                else:
                    centroids[label.item()] += emb
                    class_samples[label.item()] += 1
    
    # Compute averages and normalize
    for class_id in centroids:
        centroids[class_id] = centroids[class_id] / class_samples[class_id]
        centroids[class_id] = F.normalize(centroids[class_id], p=2, dim=0)
    
    return centroids

def find_similar_classes(query_class, centroids, k=5):
    query_centroid = centroids[query_class]
    similarities = {}
    
    for class_id, centroid in centroids.items():
        if class_id != query_class:
            sim = F.cosine_similarity(query_centroid.unsqueeze(0), 
                                    centroid.unsqueeze(0))
            similarities[class_id] = sim.item()
    
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]


# Confusion Matrix
def get_confusion_based_similarities(model, dataloader):
    # TODO get num_classes from dataloader / num lines in fonts.txt
    num_classes = 10
    confusion = torch.zeros(num_classes, num_classes)
    model.eval()
    
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            for prob, label in zip(probs, labels):
                confusion[label] += prob
    
    # Normalize by class frequency
    confusion = confusion / confusion.sum(dim=1, keepdim=True)
    return confusion

