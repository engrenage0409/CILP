"""Bidirectional retrieval evaluation for exactly one caption per image."""
import torch


@torch.no_grad()
def evaluate_retrieval(model, loader, device, ks=(1, 5)):
    model.eval()
    image_vectors, text_vectors = [], []
    for batch in loader:
        images = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        masks = batch["attention_mask"].to(device)
        image_vectors.append(model.encode_image(images).cpu())
        text_vectors.append(model.encode_text(ids, masks).cpu())

    if not image_vectors:
        raise ValueError("Validation set is empty")
    images = torch.cat(image_vectors, dim=0)  # [N, D]
    texts = torch.cat(text_vectors, dim=0)  # [N, D]
    if len(images) != len(texts):
        raise ValueError("Expected exactly one caption per image")
    sim = images @ texts.T  # [N, N] cosine similarity
    targets = torch.arange(sim.shape[0])  # diagonal is correct

    def recall(scores, k):
        k = min(k, scores.shape[1])
        topk = scores.topk(k, dim=1).indices  # [N, k]
        return (topk == targets[:, None]).any(dim=1).float().mean().item()

    metrics = {}
    for k in ks:
        metrics[f"I2T_R@{k}"] = recall(sim, k)
        metrics[f"T2I_R@{k}"] = recall(sim.T, k)
    return metrics
