# 檔案：python/pipeline/test_export_pipeline.py
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn


class Simple1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def test_onnx_export_structure_and_numerical_alignment():
    # 建立模型並確保置於評估模式
    model = Simple1DCNN()
    model.eval()

    # 指定固定輸出路徑，便於後續使用 Netron 檢驗
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "test_model.onnx"
    dummy_input = torch.randn(1, 1, 16, dtype=torch.float32)

    # 導出模型並配置輸入/輸出名稱與動態批次軸
    # 引入符號維度定義
    batch_dim = torch.export.Dim("batch_size")

    torch.onnx.export(
        model,
        (dummy_input,),
        f=str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        # 使用現代 dynamic_shapes 替代 dynamic_axes
        dynamic_shapes={
            "x": {0: batch_dim},
        },
        opset_version=18,
    )

    assert onnx_path.is_file()
    print(f"\n[Artifact Generated] ONNX model ready for Netron at: {onnx_path.resolve()}")

    # 驗證 ONNX 檔案靜態結構合法性
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 驗證動態 Batch Size 推論 (使用 Batch Size = 3)
    dynamic_test_input = torch.randn(3, 1, 16, dtype=torch.float32)

    with torch.no_grad():
        expected_output = model(dynamic_test_input).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dynamic_test_input.numpy()}

    # 執行 ONNX Runtime 推論
    actual_output = session.run(None, ort_inputs)[0]

    # 數值對齊校驗
    assert actual_output.shape == (3, 2)
    np.testing.assert_allclose(expected_output, actual_output, rtol=1e-4, atol=1e-5)

