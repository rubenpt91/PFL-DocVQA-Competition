import numpy as np
import torch

def get_bytes_for_tensor(tensor):
    """
    Computes the amount of bytes required for storing 
    the given tensor in bytes.
    """    
    if torch.is_tensor(tensor):
        return tensor.storage().nbytes()
    else:
        return tensor.nbytes