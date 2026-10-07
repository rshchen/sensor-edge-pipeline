# 檔案：python/basics/test_cifar_pipeline.py
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


class ToyDataset(Dataset):
    def __init__(self, size=8):
        self.size = size
        # 模擬 8 筆 3x32x32 假影像與標籤
        self.data = torch.zeros((size, 3, 32, 32), dtype=torch.float32)
        self.targets = torch.tensor([i % 2 for i in range(size)], dtype=torch.int64)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 32 * 32, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def test_v2_normalization_math():
    img_tensor = torch.zeros((3, 2, 2), dtype=torch.float32)
    norm = v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    output = norm(img_tensor)

    expected = torch.full((3, 2, 2), -1.0)
    assert torch.allclose(output, expected)


def test_dataloader_batch_unpacking_and_shapes():
    dataset = ToyDataset(size=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    batch_data = next(iter(loader))
    # 驗證 DataLoader 吐出的資料結構為包含 [inputs, labels] 的序列
    assert isinstance(batch_data, (list, tuple))
    assert len(batch_data) == 2

    inputs, labels = batch_data
    # 驗證 inputs 形狀為 (N, C, H, W)，labels 形狀為 (N,)
    assert inputs.shape == (4, 3, 32, 32)
    assert inputs.dtype == torch.float32
    assert labels.shape == (4,)
    assert labels.dtype == torch.int64


def test_model_state_dict_save_and_load():
    model_orig = ToyClassifier()

    with tempfile.TemporaryDirectory() as tmpdir:
        # 使用 pathlib.Path 管理路徑
        checkpoint_path = Path(tmpdir) / "model.pt"

        # 儲存模型狀態字典
        torch.save(model_orig.state_dict(), checkpoint_path)

        model_loaded = ToyClassifier()
        # 安全載入純權重並注入模型
        state_dict = torch.load(checkpoint_path, weights_only=True)
        model_loaded.load_state_dict(state_dict)

        for p1, p2 in zip(model_orig.parameters(), model_loaded.parameters()):
            assert torch.equal(p1, p2)


def test_evaluation_metric_accumulation_no_grad():
    model = ToyClassifier()
    dataset = ToyDataset(size=6)
    # batch_size=4 導致第二批次樣本數為 2（驗證不完整批次）
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    total_samples = 0
    total_correct = 0

    # 評估模式停用梯度追蹤
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            # 動態累加當前批次長度，避免受不完整批次偏差影響
            total_samples += labels.size(0)
            # 提取純量數值進行統計，切斷計算圖連結
            total_correct += (predicted == labels).sum().item()

    assert total_samples == 6
    assert isinstance(total_correct, int)


def test_accelerator_device_selection():
    device_type = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    device = torch.device(device_type)

    t = torch.randn(2, 2)
    t_device = t.to(device)

    assert t_device.device.type == device.type

