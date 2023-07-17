from differential_privacy.dp_utils import clip_parameters, flatten_params, get_shape, reconstruct_shape, add_dp_noise
import torch
import numpy as np


def test_get_shape():
    test_shapes = [(1, )]
    tensor_list = [torch.empty(test_shapes[0])]
    assert np.all(np.equal(test_shapes, get_shape(tensor_list)))

    test_shapes = [(3*i+2, 30*i) for i in range(20)]
    tensor_list = [torch.empty(test_shapes[i]) for i in range(20)]
    assert np.all(np.equal(test_shapes, get_shape(tensor_list)))


def test_flatten_params():
    test_shapes = [(3*i+2, 30*i) for i in range(20)]
    tensor_list = [torch.empty(test_shapes[i]) for i in range(20)]
    flatted_params = flatten_params(tensor_list)
    assert sum([x.numel() for x in tensor_list]) == flatted_params.numel()
    assert flatted_params.dim() == 1


def test_flatten_reconstruct_params():
    test_shapes = [(3*i+2, 30*i) for i in range(20)]
    tensor_list = [torch.randn(test_shapes[i]) for i in range(20)]
    flatted_params = flatten_params(tensor_list)
    reconstructed = reconstruct_shape(flatted_params, test_shapes)
    assert np.all([torch.equal(x, y) for x, y in zip(tensor_list, reconstructed)])
    assert np.all([np.equal(x.shape, shape) for x, shape in zip(tensor_list, test_shapes)])


def test_clip_parameters():
    test_shapes = [(i*10) for i in range(10)]
    clipping_norms = np.linspace(0, 1, num=10)
    for i, shape in enumerate(test_shapes):
        tensor = torch.rand(shape) + 20
        clipped_tensor = clip_parameters(tensor, clipping_norms[i])
        
        # compare with torch
        assert torch.linalg.vector_norm(clipped_tensor, ord=2) <= clipping_norms[i] + 1e-4

        # compare with numpy and add a bit of slack because of different precision
        assert np.linalg.norm(clipped_tensor.numpy()) <= clipping_norms[i] + 1e-4


def test_restore_frozen_weights():

    import copy
    from utils import  parse_args, load_config
    from build_utils import build_model
    args = parse_args()
    config = load_config(args)

    model = build_model(config)
    parameters = copy.deepcopy(list(model.model.state_dict().values()))

    keyed_parameters = {n: p.requires_grad for n, p in model.model.named_parameters()}
    frozen_parameters = [keyed_parameters[n] if n in keyed_parameters else True for n, p in model.model.state_dict().items()]

    agg_update = None
    new_update = [w - w_0 for w, w_0 in zip(list(model.model.state_dict().values()), parameters)]  # Get model update
    shapes = get_shape(new_update)
    new_update = flatten_params(new_update)
    new_update = clip_parameters(new_update, clip_norm=config.dp_params.sensitivity)
    agg_update = new_update

    agg_update = add_dp_noise(agg_update, noise_multiplier=config.dp_params.noise_multiplier, sensitivity=config.dp_params.sensitivity)

    # Divide the noisy aggregated update by the number of providers (Average update).
    agg_update = torch.div(agg_update, 10)

    # Add the noisy update to the original model
    agg_update = reconstruct_shape(agg_update, shapes)

    # Restore original weights (without noise) from frozen layers.
    final_update = [upd if not is_frozen else params for upd, params, is_frozen in zip(agg_update, parameters, frozen_parameters)]

    assert all([torch.all(params == new_params).item() == is_frozen for params, new_params, is_frozen in zip(parameters, final_update, frozen_parameters)])


if __name__ == '__main__':
    test_get_shape()
    test_flatten_params()
    test_flatten_reconstruct_params()
    test_clip_parameters()
    # test_restore_frozen_weights()
