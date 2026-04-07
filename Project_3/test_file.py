# A file for testing torch functions, as online compilers do not support torch.
# Has no use in the main code.

import numpy as np
import torch

# print(5 % 3)
# print(2 % 3)
# print(torch.arange(0, 5) % 3)

# action = 3

# action_indice = torch.zeros([1,4])
# action_indice[0][action] = torch.tensor(1) # assuming action is an integer from 0-3
        

# print(action_indice)

a = torch.randn(4, 4)
print(a)
print(torch.sum(a, 0))
b = torch.arange(4 * 5 * 6).view(4, 5, 6)
torch.sum(b, (2, 1))