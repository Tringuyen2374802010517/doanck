import numpy as np
import torch

def predict_image(model, image, embeddings, labels, class_names, full_labels, top_k=3):
    model.eval()

    with torch.no_grad():
        emb = model(image)
        emb = emb.cpu().numpy()

    emb = emb / np.linalg.norm(emb)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    sims = embeddings_norm @ emb.T
    sims = sims.squeeze()

    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "pill_code": str(class_names[labels[idx]]),
            "full_label": str(full_labels[idx]),
            "score": round(float(sims[idx]) * 100, 2),
            "image_path": f"/static/database/{class_names[labels[idx]]}.jpg"  # 🔥 FIX
        })

    return results