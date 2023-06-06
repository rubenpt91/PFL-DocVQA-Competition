from communication.compute_tensor_size import get_bytes_for_tensor
import torch

def test_get_bytes_for_tensor_float64():
    b = torch.randn(128, 256, dtype=torch.float64)
    amount_for_float64 = 8
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_float64

def test_get_bytes_for_tensor_float32():
    b = torch.randn(1, 1, 128, 256, dtype=torch.float32)
    amount_for_float32 = 4
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_float32

def test_get_bytes_for_tensor_float16():
    b = torch.randn(1, 1, 128, 256, dtype=torch.float16)
    amount_for_float16 = 2
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_float16

def test_get_bytes_for_tensor_int8():
    b = torch.empty(1, 1, 128, 256, dtype=torch.int8)
    amount_for_int8 = 1
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_int8

def test_get_bytes_for_tensor_bool():
    """
    Weirdly enough one bool takes 1 Byte in torch.
    See: https://github.com/pytorch/pytorch/issues/41571
    """
    b = torch.empty(1, 1, 128, 256, dtype=torch.bool)
    amount_for_bool = 1
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_bool

def test_equal_get_bytes_for_tensor():
    b = torch.randn(1, 1, 128, 256, dtype=torch.float64)
    c = torch.randn(1, 1, 128, 256, dtype=torch.float64)
    assert get_bytes_for_tensor(b) == get_bytes_for_tensor(c)
