// 檔案：cpp/main.cpp
#include <iostream>
#include "onnxruntime_cxx_api.h"

int main() {
    std::cout << "Testing ONNX Runtime headers and linkage..." << std::endl;

    // 取得 ONNX Runtime 版本字串以驗證動態庫符號連結與執行期載入
    const char* version = OrtGetApiBase()->GetVersionString();
    std::cout << "ONNX Runtime Version: " << version << std::endl;

    // 建立基礎環境物件以驗證 C++ RAII 封裝層
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SensorEdgePipeline");
    std::cout << "Ort::Env successfully initialized." << std::endl;

    return 0;
}

