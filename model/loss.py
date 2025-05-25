import torch

def relative_l2_loss(x, y):
    b = x.shape[0]
    diff_norms = torch.norm(x.reshape(b, -1) - y.reshape(b, -1), p = 2, dim = -1)
    y_norms = torch.norm(y.reshape(b, -1), p = 2, dim = -1)
    return torch.mean(diff_norms / y_norms)


"""
the caculation of mse loss is different in mgn and ggns
refer to
mgn: https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/meshgraphnets/cloth_model.py#L88
ggns: https://github.com/jlinki/GGNS/blob/69c5efac311aa113aed799bfda055650bc70cad4/evaluate_model.py#L251
in our experiment
we use ggns-type loss in rollout experiments
"""
def mse_l2_loss(x, y, loss_resource = "ggns", use_sqrt = False):
    if loss_resource == "mgn":
        loss = torch.mean(torch.sum((x - y) ** 2, dim = -1))
    elif loss_resource == "ggns":
        loss = torch.nn.MSELoss()(x, y)
    if use_sqrt:
        loss = torch.sqrt(loss)
    return loss