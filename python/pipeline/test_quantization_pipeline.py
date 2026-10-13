# 檔案：python/pipeline/test_quantization_pipeline.py
from pathlib import Path
import numpy as np
import pytest
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
    CalibrationMethod,
    QuantFormat,
    QuantType
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class SensorCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_data: np.ndarray, input_name: str = "input"):
        self.data = calibration_data
        self.input_name = input_name
        self.total_samples = calibration_data.shape[0]
        self.current_idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        # 當資料讀取完畢時終止迭代
        if self.current_idx >= self.total_samples:
            return None
        
        # 確保提取樣本記憶體連續且資料型別為 float32
        sample = np.ascontiguousarray(
            self.data[self.current_idx : self.current_idx + 1], 
            dtype=np.float32
        )
        self.current_idx += 1
        return {self.input_name: sample}

    def rewind(self) -> None:
        self.current_idx = 0


def compute_metrics(y_fp32: np.ndarray, y_int8: np.ndarray) -> dict[str, float]:
    """計算 MAE, MSE 與 SNR 指標"""
    diff = y_fp32 - y_int8
    mae = float(np.max(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    
    signal_power = np.sum(y_fp32 ** 2)
    noise_power = np.sum(diff ** 2)
    
    # 計算以 dB 為單位的訊噪比
    snr = 10.0 * np.log10(signal_power / (noise_power + 1e-12))
    return {"MAE": mae, "MSE": mse, "SNR_dB": float(snr)}


def test_static_quantization_pipeline():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_model_path = project_root / "models" / "test_model.onnx"
    preprocessed_model_path = project_root / "models" / "preprocessed_model.onnx"
    quantized_model_path = project_root / "models" / "quantized_model.onnx"

    # 1. 執行顯式前處理與符號形狀推導
    quant_pre_process(
        input_model=str(raw_model_path),
        output_model_path=str(preprocessed_model_path),
        skip_optimization=False,
        skip_symbolic_shape=False
    )
    assert preprocessed_model_path.exists()

    # 2. 準備校準資料並實作 DataReader
    calib_samples = np.random.uniform(-1.0, 1.0, size=(16, 1, 16)).astype(np.float32)
    data_reader = SensorCalibrationDataReader(calib_samples, input_name="input")

    # 3. 執行後訓練靜態量化 (PTQ)
    quantize_static(
        model_input=str(preprocessed_model_path),
        model_output=str(quantized_model_path),
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Entropy
    )
    assert quantized_model_path.exists()

    # 4. 驗證 FP32 與 INT8 輸出對齊與精度度量
    test_input = np.ones((1, 1, 16), dtype=np.float32)
    
    session_fp32 = ort.InferenceSession(str(raw_model_path))
    out_fp32 = session_fp32.run(None, {"input": test_input})[0]

    session_int8 = ort.InferenceSession(str(quantized_model_path))
    out_int8 = session_int8.run(None, {"input": test_input})[0]

    metrics = compute_metrics(out_fp32, out_int8)
    assert metrics["SNR_dB"] > 25.0

