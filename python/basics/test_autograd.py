# 檔案：python/basics/test_autograd.py
import torch


def test_autograd_graph_attributes():
    x = torch.ones((2, 2), requires_grad=True)
    y = x + 2

    # 1. 檢驗葉節點與計算圖歷史連結
    assert x.is_leaf is True
    assert y.is_leaf is False
    assert y.grad_fn is not None


def test_scalar_backward_computation():
    x = torch.tensor(3.0, requires_grad=True)
    # y = 2x^2 + 1, dy/dx = 4x -> x=3 時導數為 12.0
    y = 2 * (x ** 2) + 1

    # 2. 觸發反向傳播求導
    y.backward()

    expected_grad = torch.tensor(12.0)
    assert torch.equal(x.grad, expected_grad)


def test_no_grad_context():
    x = torch.ones((2, 2), requires_grad=True)

    # 3. 停用梯度追蹤區塊
    with torch.no_grad():
        y = x * 2

    assert y.requires_grad is False
    assert y.grad_fn is None


def test_vector_jacobian_product():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = torch.empty(2)
    y[0] = x[0] * 2
    y[1] = x[1] * 3

    # 4. 非純量求導傳入權重向量 v = [1.0, 1.0]
    v = torch.tensor([1.0, 1.0])
    y.backward(v)

    expected_grad = torch.tensor([2.0, 3.0])
    assert torch.equal(x.grad, expected_grad)


def test_gradient_accumulation_and_zero():
    w = torch.tensor(2.0, requires_grad=True)

    loss1 = w * 3
    loss1.backward()
    assert torch.equal(w.grad, torch.tensor(3.0))

    # 5. 連續計算產生累積效應
    loss2 = w * 4
    loss2.backward()
    assert torch.equal(w.grad, torch.tensor(7.0))

    # 6. 就地清空梯度
    w.grad.zero_()
    assert torch.equal(w.grad, torch.tensor(0.0))


def test_higher_order_derivative():
    x = torch.tensor(2.0, requires_grad=True)
    # y = 5x^3, dy/dx = 15x^2, d^2y/dx^2 = 30x -> x=2 時二階導數為 60.0
    y = 5 * (x ** 3)

    # 7. 顯式計算一階導函數並建立反向傳播計算圖
    grad_x = torch.autograd.grad(outputs=y, inputs=x, create_graph=True)[0]
    assert grad_x.grad_fn is not None

    # 8. 對一階導函數進行反向傳播
    grad_x.backward()

    expected_second_order_grad = torch.tensor(60.0)
    assert torch.equal(x.grad, expected_second_order_grad)

