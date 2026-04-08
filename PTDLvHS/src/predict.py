import numpy as np
import torch

def predict_image(model, image, embeddings, labels, class_names, full_labels, top_k=3):
    with torch.no_grad():
        emb = model(image).cpu().numpy()

    sims = embeddings @ emb.T
    sims = sims.squeeze()

    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "pill_code": str(class_names[labels[idx]]),   # mã thuốc ngắn
            "full_label": str(full_labels[idx]),          # mã đầy đủ
            "score": round(float(sims[idx]) * 100, 2)
        })

    return results