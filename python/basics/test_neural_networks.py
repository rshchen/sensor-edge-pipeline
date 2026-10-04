# 檔案：python/basics/test_neural_networks.py
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_module_parameter_registration():
    model = SimpleMLP()
    
    # 1. 取得模型內所有參數列表
    params = list(model.parameters())
    
    # 2 層網路，每層包含 weight 與 bias，共 4 個 Parameter
    assert len(params) == 4
    # 驗證首層權重形狀為 (輸出維度, 輸入維度)
    assert params[0].shape == (8, 4)


def test_batch_dimension_alignment():
    model = SimpleMLP()
    single_sample = torch.randn(4)  # 形狀為 (4,)，缺少 Batch 維度

    # 2. 在維度 0 插入批次軸
    batched_sample = single_sample.unsqueeze(0)
    assert batched_sample.shape == (1, 4)

    output = model(batched_sample)
    assert output.shape == (1, 2)


def test_training_step_optimization():
    torch.manual_seed(42)
    model = SimpleMLP()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 2)

    # 備份原始權重數值複本
    initial_weight = model.fc1.weight.clone()

    # 3. 執行標準單步訓練流程
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    # 驗證權重數值已被更新
    assert not torch.equal(model.fc1.weight, initial_weight)


def test_gradient_zeroing_behavior():
    model = SimpleMLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    inputs = torch.randn(1, 4)
    targets = torch.randn(1, 2)

    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()

    # 驗證反向傳播後已累積梯度
    assert model.fc1.weight.grad is not None
    assert not torch.all(model.fc1.weight.grad == 0.0)

    # 4. 清除最佳化器追蹤之所有梯度
    optimizer.zero_grad()

    # 驗證梯度緩衝區已被清空或重置為 0
    assert (model.fc1.weight.grad is None) or torch.all(model.fc1.weight.grad == 0.0)

