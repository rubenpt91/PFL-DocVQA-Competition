from communication.compute_tensor_size import get_bytes_for_tensor
import numpy as np

def test_get_bytes_for_tensor_float64():
    b = np.empty((128, 256), dtype=np.float64)
    amount_for_float64 = 8
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_float64

def test_get_bytes_for_tensor_float32():
    b = np.empty((1, 1, 128, 256), dtype=np.float32)
    amount_for_float32 = 4
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_float32

def test_get_bytes_for_tensor_float16():
    b = np.empty((1, 1, 128, 256), dtype=np.float16)
    amount_for_float16 = 2
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_float16

def test_get_bytes_for_tensor_int8():
    b = np.empty((1, 1, 128, 256), dtype=np.int8)
    amount_for_int8 = 1
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_int8

def test_get_bytes_for_tensor_bool():
    """
    Weirdly enough one bool takes 1 Byte in np and torch.
    See numpy: https://github.com/numpy/numpy/issues/14821
    See torch: https://github.com/pynp/pynp/issues/41571
    """
    b = np.empty((1, 1, 128, 256), dtype=bool)
    amount_for_bool = 1
    assert get_bytes_for_tensor(b) == 128 * 256 * amount_for_bool

def test_equal_get_bytes_for_tensor():
    b = np.empty((1, 1, 128, 256), dtype=np.float64)
    c = np.empty((1, 1, 128, 256), dtype=np.float64)
    assert get_bytes_for_tensor(b) == get_bytes_for_tensor(c)
