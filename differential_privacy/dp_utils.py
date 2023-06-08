import torch

def flatten_params(update):
    """
    Flat the list of tensors (layer params) into a single vector.
    """
    # return np.concatenate([np.array(element).ravel() for element in update])
    return torch.cat([torch.flatten(element) for element in update])

def compute_norm(a):
    """
    Compute L2 norm of a vector.
    """
    # return LA.norm(a, ord=2)
    return torch.linalg.vector_norm(a, ord=2)

def clip_norm(a,clip_norm):
    """
    Clip update parameters to clip norm.
    """
    # return np.divide(a, np.maximum(1, np.divide(compute_norm(a), clip_norm)))
    return torch.div(a, torch.max(torch.tensor(1, device=a.device), torch.div(compute_norm(a), clip_norm)))

def get_shape(update):
    """
    Get the shapes of the tensors to be reconstructed later.
    """
    shapes=[ele.shape for ele in update]
    return shapes

def reconstruct(flat_update,shapes):
    """
    Reconstruct the original shapes of the tensors list.
    """
    ind=0
    rec_upd=[]
    for shape in shapes:
        num_elements = torch.prod(torch.tensor(shape)).item()
        rec_upd.append(flat_update[ind:ind+num_elements].reshape(shape))
        ind+=num_elements

    return rec_upd