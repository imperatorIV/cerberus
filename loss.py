import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F



#Function modified to work with shared_embeddings (coming as an output from the cross attention module)

def contrastive_loss(shared_embeddings):
    """
    Calculate contrastive loss based on shared embeddings.

    Args:
    - shared_embeddings (torch.Tensor): Shared embeddings from cross-attention module, shape (batch_size, embedding_size).

    Returns:
    - torch.Tensor: Contrastive loss.
    """

    pairwise_similarity = torch.matmul(shared_embeddings, shared_embeddings.t())

    # Create labels (1 for positive pairs, 0 for negative pairs)
    labels = torch.arange(len(shared_embeddings)).to(shared_embeddings.device)

    loss = F.cross_entropy(pairwise_similarity, labels)

    return loss


#The orignal function was taken from this repo : https://github.com/filipbasara0/simple-clip/blob/afa7b954fedde5aa51bad2913ffecc370a7164fd/simple_clip/clip.py#L9
