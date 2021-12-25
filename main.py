import torch
import torch.nn.functional as F

if __name__ == '__main__':
    t = torch.tensor([[1, 2], [2, 1], [2, 1], [3, 6]], dtype=torch.float)
    t = F.softmax(t, dim=1)
    print(t)

