import torch
from collections import OrderedDict


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.model.load_state_dict(state_dict, strict=True)


"""
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
"""

import warnings
def weighted_average(metrics_dict):
    warnings.warn('\n' + str(metrics_dict) + '\n')

    metrics = list(metrics_dict[0][1].keys())
    aggregated_metrics_dict = {}
    dataset_length = sum([num_samples for num_samples, _ in metrics_dict])
    for metric in metrics:
        aggregated_metrics_dict[metric] = sum(
            [m[metric] * num_examples / dataset_length for num_examples, m in metrics_dict])

    warnings.warn('\n' + str(metrics_dict) + '\n')
    return aggregated_metrics_dict
