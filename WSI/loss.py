import torch


# AUXILIARY METHODS
def calculate_classification_error(Y_hat, Y):
    Y = Y.float()
    error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

    return error

def calculate_objective(Y_prob, Y):
    Y = Y.float()
    Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
    neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

    return neg_log_likelihood