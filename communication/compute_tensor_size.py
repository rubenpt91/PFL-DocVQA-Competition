import torch

def get_bytes_for_tensor(tensor : torch.Tensor):
    """
    Computes the amount of bytes required for storing 
    the given tensor in bytes.
    """    
    return tensor.nbytes