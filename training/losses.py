import torch
import torch.nn.functional as F

def MNRL_loss(embedding1, embedding2, eps=1e-8, device='cuda'):
    embedding1_norm = embedding1 / (eps + embedding1.norm(dim=1, keepdim=True))
    embedding2_norm = embedding2 / (eps + embedding2.norm(dim=1, keepdim=True))
    similarity_matrix = embedding1_norm @ embedding2_norm.T
    return F.cross_entropy(similarity_matrix + eps, torch.arange(similarity_matrix.shape[0], device=device))


def MRR(embedding1, embedding2, eps=1e-8, device=None):
    if device is None:
        device = embedding1.device
    embedding1_norm = embedding1 / (eps + embedding1.norm(dim=1, keepdim=True))
    embedding2_norm = embedding2 / (eps + embedding2.norm(dim=1, keepdim=True))
    similarity_matrix = embedding1_norm @ embedding2_norm.T
    _, ranks = similarity_matrix.sort(descending=True, dim=-1)
    positions = torch.nonzero(ranks == torch.arange(similarity_matrix.shape[0], device=device).view(-1, 1))[:, 1]
    return (1 / (positions + 1)).mean() 