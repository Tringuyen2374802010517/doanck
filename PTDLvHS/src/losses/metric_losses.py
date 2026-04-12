import torch.nn as nn

triplet_loss = nn.TripletMarginLoss(margin=0.5)
cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)


def compute_total_loss(a_emb, p_emb, n_emb, logits, labels):
    loss1 = triplet_loss(a_emb, p_emb, n_emb)
    loss2 = cls_loss(logits, labels)

    total = loss1 + 0.5 * loss2
    return total, loss1, loss2