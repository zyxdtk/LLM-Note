import torch

x = torch.tensor([1., 2., 3.])
result = torch.logsumexp(x, dim=0)
print(result)


def my_logsumexp(x, dim=None, keepdim=True):
    mask_value = torch.tensor(-float('inf'), dtype=x.dtype, device=x.device)
    x_masked = torch.where(x < -1000, mask_value, x)
    if dim is None:
        max_val = torch.max(x_masked)
    else:
        max_val = torch.max(x_masked, dim=dim, keepdim=keepdim).values
    max_val_safe = torch.where(torch.isfinite(max_val), max_val, torch.zeros_like(max_val))
    exp_shifted = torch.exp(x - max_val_safe)
    sum_exp = torch.sum(exp_shifted, dim=dim, keepdim=keepdim)
    if not keepdim:
        max_val_safe = max_val_safe.squeeze(dim)
    return max_val_safe + torch.log(sum_exp)


result2 = my_logsumexp(x)
print(result2)