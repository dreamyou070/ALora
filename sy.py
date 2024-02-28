import torch
from torch import nn

def right_rotate(test_list, n):
  new_list = [None for _ in range(len(test_list))] # [None, None, None, None, None]
  for i in range(len(test_list)):
      new_list[i] = test_list[i-n]
  return new_list


a_list = [3,4,5,1]
b_list = right_rotate(a_list, 1)
print(b_list)