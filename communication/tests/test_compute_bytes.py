from communication.compute_tensor_size import get_bytes_for_tensor
import numpy as np
import torch

def test_get_bytes_for_tensor_float64_np():
    b = np.empty((128, 256), dtype=np.float64)
    amount_for_float64 = 8
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_float64

def test_get_bytes_for_tensor_float32_np():
    b = np.empty((1, 1, 128, 256), dtype=np.float32)
    amount_for_float32 = 4
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_float32

def test_get_bytes_for_tensor_float16_np():
    b = np.empty((1, 1, 128, 256), dtype=np.float16)
    amount_for_float16 = 2
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_float16

def test_get_bytes_for_tensor_int8_np():
    b = np.empty((1, 1, 128, 256), dtype=np.int8)
    amount_for_int8 = 1
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_int8

def test_get_bytes_for_tensor_bool_np():
    """
    Weirdly enough one bool takes 1 Byte in np.
    See numpy: https://github.com/numpy/numpy/issues/14821
    """
    b = np.empty((1, 1, 128, 256), dtype=bool)
    amount_for_bool = 1
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_bool

def test_equal_get_bytes_for_tensor_np():
    b = np.empty((1, 1, 128, 256), dtype=np.float64)
    c = np.empty((1, 1, 128, 256), dtype=np.float64)
    assert get_bytes_for_tensor(b) == get_bytes_for_tensor(c)

def test_get_bytes_for_tensor_float64_torch():
    b = torch.randn(128, 256, dtype=torch.float64)
    amount_for_float64 = 8
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_float64

def test_get_bytes_for_tensor_float32_torch():
    b = torch.randn(1, 1, 128, 256, dtype=torch.float32)
    amount_for_float32 = 4
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_float32

def test_get_bytes_for_tensor_float16_torch():
    b = torch.randn(1, 1, 128, 256, dtype=torch.float16)
    amount_for_float16 = 2
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_float16

def test_get_bytes_for_tensor_int8_torch():
    b = torch.empty(1, 1, 128, 256, dtype=torch.int8)
    amount_for_int8 = 1
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_int8

def test_get_bytes_for_tensor_bool_torch():
    """
    Weirdly enough one bool takes 1 Byte in torch.
    See: https://github.com/pytorch/pytorch/issues/41571
    """
    b = torch.empty(1, 1, 128, 256, dtype=torch.bool)
    amount_for_bool = 1
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_bool

def test_equal_get_bytes_for_tensor_torch():
    b = torch.randn(1, 1, 128, 256, dtype=torch.float64)
    c = torch.randn(1, 1, 128, 256, dtype=torch.float64)
    assert get_bytes_for_tensor(b) == get_bytes_for_tensor(c)