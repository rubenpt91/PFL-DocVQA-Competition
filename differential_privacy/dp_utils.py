import torch


def flatten_params(parameters):
    """
    Flat the list of tensors (layer params) into a single vector.
    """
    return torch.cat([torch.flatten(layer_norm) for layer_norm in parameters])


def clip_parameters(parameters, clip_norm):
    """
    Clip update parameters to clip norm.
    """
    current_norm = torch.linalg.vector_norm(parameters, ord=2)
    return torch.div(parameters, torch.max(torch.tensor(1, device=parameters.device), torch.div(current_norm, clip_norm)))


def get_shape(update: list):
    """
    Return a list of shapes given a list of tensors.
    """
    shapes = [ele.shape for ele in update]
    return shapes


def reconstruct_shape(flat_update, shapes):
    """
    Reconstruct the original shapes of the tensors list.
    """
    ind = 0
    rec_upd = []
    for shape in shapes:
        num_elements = torch.prod(torch.tensor(shape)).item()
        rec_upd.append(flat_update[ind:ind+num_elements].reshape(shape))
        ind += num_elements

    return rec_upd


def add_dp_noise(data, noise_multiplier, sensitivity):
    """
    Add differential privacy noise to data.
    """
    return torch.add(data, torch.normal(mean=0, std=noise_multiplier * sensitivity, size=data.shape, device=data.device))
