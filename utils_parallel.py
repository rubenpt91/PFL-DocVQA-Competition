import torch
from collections import OrderedDict


def get_parameters_from_model(model):

    get_params_func = getattr(model, "get_model_parameters", None)
    if callable(get_params_func):
        params = model.get_model_parameters()

    else:
        params = [val.cpu().numpy() for _, val in model.model.state_dict().items()]

    return params


def set_parameters_model(model, parameters):
    set_params_func = getattr(model, "set_model_parameters", None)
    if callable(set_params_func):
        model.set_model_parameters(parameters)

    else:
        params_dict = zip(model.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.model.load_state_dict(state_dict, strict=True)


def weighted_average(metrics_dict):
    metrics = list(metrics_dict[0][1].keys())
    aggregated_metrics_dict = {}
    dataset_length = sum([num_samples for num_samples, _ in metrics_dict])
    for metric in metrics:
        aggregated_metrics_dict[metric] = sum(
            [m[metric] * num_examples / dataset_length for num_examples, m in metrics_dict])

    return aggregated_metrics_dict
