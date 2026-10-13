// 檔案：cpp/quantized_infer.cpp
#include <iostream>
#include <vector>
#include "onnxruntime_cxx_api.h"

int main() {
    std::cout << "=== Edge C++ INT8 Model Inference ===" << std::endl;

    // 1. 初始化全域環境實例與會話選項
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SensorInferEngine");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(2);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 2. 載入量化後的模型
    const std::string model_path = "../models/quantized_model.onnx";
    Ort::Session session(env, model_path.c_str(), session_options);

    // 3. 準備連續輸入緩衝區（形狀：1x1x16）
    const std::vector<int64_t> input_shape = {1, 1, 16};
    const size_t input_elements = 1 * 1 * 16;
    std::vector<float> input_buffer(input_elements, 0.5f);

    // 4. 配置 CPU 記憶體資訊描述
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, 
        OrtMemType::OrtMemTypeDefault
    );

    // 5. 原生記憶體零拷貝封裝為張量
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_buffer.data(),
        input_buffer.size(),
        input_shape.data(),
        input_shape.size()
    );

    // 6. 設定輸入與輸出節點名稱
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    // 7. 執行單次同步前向推論
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    // 8. 驗證並讀取輸出指標
    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    std::cout << "Forward Inference Success. First Output Value: " << output_data[0] << std::endl;

    return 0;
}

