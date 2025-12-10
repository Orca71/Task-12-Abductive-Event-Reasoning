import torch
from aer.losses import bradley_terry_loss

s_pos = torch.tensor([1.2])
s_neg = torch.tensor([-0.3])

loss = bradley_terry_loss(s_pos, s_neg)
print("Loss:", loss.item())
