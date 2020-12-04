import torch


def entropy(p, eps=1e-6):
    p = torch.clamp(p, eps, 1.0-eps)
    return -1.0*((p*torch.log(p)) + ((1.0-p)*(torch.log(1.0-p))))

def expected_entropy(mc_preds):
    return torch.mean(entropy(mc_preds), dim=0)

def predictive_entropy(mc_preds):
    return entropy(torch.mean(mc_preds, dim=0))

def uncertainty_filtering(prediction, uncertainty, tu=0.0125):

    filtered_pred = torch.zeros_like(prediction)

    for inst_id in torch.unique(prediction):
        if inst_id == 0:
            continue
        inst_mask = prediction == inst_id
        inst_uncertainty = torch.mean(uncertainty[inst_mask])
        if inst_uncertainty < tu:
            filtered_pred[inst_mask] = torch.max(filtered_pred) + 1

    return filtered_pred
