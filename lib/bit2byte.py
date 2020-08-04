"""
Accessor functions for the bit2byte library. Used to encode a float32 array containing signs (-1/1)
into a condensed representation.
Taken from https://github.com/PermiJW/signSGD-with-Majority-Vote
"""
import torch


try:
    import bit2byte
except ImportError:
    pass


def packing(src_tensor):
    src_tensor = torch.sign(src_tensor)
    src_tensor_size = src_tensor.size()
    src_tensor = src_tensor.view(-1)
    src_len = len(src_tensor)
    add_elm = 32 - (src_len % 32)
    if src_len % 32 == 0:
        add_elm = 0
    new_tensor = torch.zeros([add_elm], dtype=torch.float32, device=src_tensor.device)
    src_tensor = torch.cat((src_tensor, new_tensor), 0)
    src_tensor = src_tensor.view(32, -1)
    src_tensor = src_tensor.to(dtype=torch.int32)
    dst_tensor = bit2byte.packing(src_tensor)
    dst_tensor = dst_tensor.to(dtype=torch.int32)
    return dst_tensor, src_tensor_size


def unpacking(src_tensor, src_tensor_size):
    src_tensor = src_tensor.clone()  # edited so it doesn't destroy the input
    src_element_num = element_num(src_tensor_size)
    add_elm = 32 - (src_element_num % 32)
    if src_element_num % 32 == 0:
        add_elm = 0
    src_tensor = src_tensor.int()
    new_tensor = torch.ones(src_element_num + add_elm, device=src_tensor.device, dtype=torch.int32)
    new_tensor = new_tensor.view(32, -1)
    new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
    new_tensor = new_tensor.view(-1)
    new_tensor = new_tensor[:src_element_num]
    new_tensor = new_tensor.view(src_tensor_size)
    new_tensor = -new_tensor.add_(-1)
    new_tensor = new_tensor.float()
    return new_tensor


def element_num(size):
    num = 1
    for i in range(len(size)):
        num *= size[i]
    return num
