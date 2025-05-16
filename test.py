import torch
import square_add

a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
b = torch.tensor([4.0, 5.0, 6.0], device="cuda")

out = square_add.square_add(a, b)
print("Output:", out)