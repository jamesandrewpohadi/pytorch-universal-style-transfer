import torch
import torch.nn as nn
from .general import calc_mean_std

mse_loss = nn.MSELoss()

def content_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    return mse_loss(input, target)

def mean_std_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(input_mean, target_mean) + mse_loss(input_std, target_std)

def gram_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    size = input.size()
    assert (len(size) == 4)
    N, C = size[:2]
    flattened_input = input.view(N,C,-1)
    flattened_target = target.view(N,C,-1)
    gram_input = torch.bmm(flattened_input,flattened_input.permute(0,2,1))
    gram_target = torch.bmm(flattened_target,flattened_target.permute(0,2,1))
    return mse_loss(gram_input,gram_target)