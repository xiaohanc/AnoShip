import torch
import torch.nn.functional as F

def kl_divergence_contributions(p, q):

    window_size = p.shape[1]
    # Apply softplus function to handle negative values in p
    p = F.softplus(p)
    q = F.softplus(q)

    p_sum = torch.sum(p, dim=1).unsqueeze(1)
    p_sum = p_sum.repeat(1, window_size, 1)
    p = p / p_sum
    q_sum = torch.sum(q, dim=1).unsqueeze(1)
    q_sum = q_sum.repeat(1, window_size, 1)
    q = q / q_sum

    # Compute contributions to KL divergence
    # contributions = p * (p.log() - q.log())
    contributions = p * (torch.log(p) - torch.log(q)) * torch.log(1 + (torch.log(p) - torch.log(q)) ** 2)

    return contributions