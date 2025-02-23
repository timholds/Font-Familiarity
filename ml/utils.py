# Create a model name
import os
from pathlib import Path
def get_model_path(base_dir, prefix, batch_size, embedding_dim, initial_channels):
    model_name = f"{prefix}_BS{batch_size}-ED{embedding_dim}-IC{initial_channels}.pt"
    model_path = Path(os.path.join(base_dir, model_name))
    return model_path


def get_embedding_path(base_dir, embedding_file=None, embedding_dim=None):
    if embedding_file is not None:
        return Path(os.path.join(base_dir, embedding_file))
    else:
        # TODO make sure this still works when embedding_dim is None
        embedding_file = f"class_embeddings_{embedding_dim}.npy"
        return Path(os.path.join(base_dir, embedding_file))