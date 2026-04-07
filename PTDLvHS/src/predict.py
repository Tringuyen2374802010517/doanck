import numpy as np
import torch

def predict_image(model, image, embeddings, labels, class_names):
    with torch.no_grad():
        emb = model(image).numpy()

    sims = embeddings @ emb.T
    idx = np.argmax(sims)

    return class_names[labels[idx]]