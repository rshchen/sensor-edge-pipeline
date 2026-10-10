// 檔案：cpp/main.cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>
#include "onnxruntime_cxx_api.h"

int main() {
    std::cout << "=== Edge C++ Inference Engine & Latency Profiling ===" << std::endl;

    // 1. 初始化全域環境實例與會話選項
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SensorInferEngine");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(2);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 2. 載入模型並建立 Session
    const std::string model_path = "../models/test_model.onnx";
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

    // 7. 硬體預熱迴圈（排除冷啟動與快取失效延遲）
    const int warmup_runs = 10;
    for (int i = 0; i < warmup_runs; ++i) {
        auto warmup_output = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );
    }

    // 8. 基準測試與單調時鐘耗時度量
    const int benchmark_runs = 100;
    std::vector<double> latencies_us;
    latencies_us.reserve(benchmark_runs);

    for (int i = 0; i < benchmark_runs; ++i) {
        auto start = std::chrono::steady_clock::now();

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        auto end = std::chrono::steady_clock::now();

        double elapsed = std::chrono::duration<double, std::micro>(end - start).count();
        latencies_us.push_back(elapsed);
    }

    // 9. 統計指標計算
    double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    double mean_latency = sum / benchmark_runs;

    double variance_acc = 0.0;
    for (double val : latencies_us) {
        variance_acc += (val - mean_latency) * (val - mean_latency);
    }
    double std_dev = std::sqrt(variance_acc / benchmark_runs);

    std::cout << "Mean Forward Latency : " << mean_latency << " us (" << mean_latency / 1000.0 << " ms)" << std::endl;
    std::cout << "Latency Jitter (Std) : " << std_dev << " us (" << std_dev / 1000.0 << " ms)" << std::endl;

    return 0;
}

