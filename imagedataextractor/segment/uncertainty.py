"""
Functions for computing conditional mutual information, uncertainty and uncertainty filtering.

.. codeauthor:: Batuhan Yildirim <by256@cam.ac.uk>
"""

import torch


def entropy(p, eps=1e-6):
    p = torch.clamp(p, eps, 1.0-eps)
    return -1.0*((p*torch.log(p)) + ((1.0-p)*(torch.log(1.0-p))))

def expected_entropy(mc_preds):
    """Aleatoric (data) uncertainty"""
    return torch.mean(entropy(mc_preds), dim=0)

def predictive_entropy(mc_preds):
    """Total uncertainty"""
    return entropy(torch.mean(mc_preds, dim=0))

def uncertainty_filtering(prediction, uncertainty, tu=0.0125):
    """Filters instance segmentaton predictions based on their uncertainty."""

    filtered_pred = torch.zeros_like(prediction)

    for inst_id in torch.unique(prediction):
        if inst_id == 0:
            continue
        inst_mask = prediction == inst_id
        inst_uncertainty = torch.mean(uncertainty[inst_mask])
        if inst_uncertainty < tu:
            filtered_pred[inst_mask] = torch.max(filtered_pred) + 1

    return filtered_pred
