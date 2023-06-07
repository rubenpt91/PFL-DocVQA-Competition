import numpy as np

def get_bytes_for_tensor(tensor : np.ndarray):
    """
    Computes the amount of bytes required for storing 
    the given tensor in bytes.
    """    
    return tensor.nbytes