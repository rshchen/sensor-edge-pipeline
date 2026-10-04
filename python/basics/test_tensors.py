# 檔案：python/basics/test_tensors.py
import numpy as np
import torch


def test_tensor_factory_and_like():
    x = torch.zeros((2, 3), dtype=torch.float32)

    # 1. 依據 x 的維度與型態建立全 1 張量
    ones_x = torch.ones_like(x)

    assert ones_x.shape == (2, 3)
    assert ones_x.dtype == torch.float32
    assert torch.all(ones_x == 1.0)


def test_tensor_concatenation():
    t = torch.ones((2, 2))

    # 2. 沿欄位方向（dim=1）拼接 3 個張量
    cat_t = torch.cat([t, t, t], dim=1)

    assert cat_t.shape == (2, 6)
    assert torch.all(cat_t == 1.0)


def test_tensor_multiplication_mechanics():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[2.0, 0.0], [1.0, 2.0]])

    # 3. 逐元素相乘 (Hadamard product)
    elem_mul = A * B
    expected_elem = torch.tensor([[2.0, 0.0], [3.0, 8.0]])
    assert torch.equal(elem_mul, expected_elem)

    # 4. 線性代數矩陣乘法
    matmul_res = A @ B
    expected_matmul = torch.tensor([[4.0, 4.0], [10.0, 8.0]])
    assert torch.equal(matmul_res, expected_matmul)


def test_numpy_zero_copy_bridge():
    np_arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    # 5. 零拷貝轉換為 PyTorch Tensor
    t = torch.from_numpy(np_arr)

    # In-place 修改 NumPy 陣列
    np.add(np_arr, 5.0, out=np_arr)

    # 驗證 PyTorch Tensor 同步改變
    expected_t = torch.tensor([15.0, 25.0, 35.0], dtype=torch.float32)
    assert torch.equal(t, expected_t)


def test_inplace_operation():
    x = torch.ones((2, 2))

    # 6. 呼叫就地加法操作
    x.add_(3.0)

    expected = torch.tensor([[4.0, 4.0], [4.0, 4.0]])
    assert torch.equal(x, expected)

