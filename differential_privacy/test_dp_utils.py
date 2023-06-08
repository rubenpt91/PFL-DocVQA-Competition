from differential_privacy.dp_utils import clip_parameters, flatten_params, get_shape, reconstruct_shape
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

