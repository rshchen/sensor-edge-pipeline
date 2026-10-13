# 檔案：cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(SensorEdgeInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 定義預編譯 ONNX Runtime 本地路徑
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")

# 建立外部匯入庫目標
add_library(onnxruntime SHARED IMPORTED)

if(APPLE)
    set(ORT_LIB_PATH "${ORT_ROOT}/lib/libonnxruntime.dylib")
    set(LOADER_ORIGIN "@loader_path")
else()
    set(ORT_LIB_PATH "${ORT_ROOT}/lib/libonnxruntime.so")
    set(LOADER_ORIGIN "$ORIGIN")
endif()

set_target_properties(onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
    IMPORTED_LOCATION "${ORT_LIB_PATH}"
)

# 階段一：連結性測試執行檔
add_executable(test_link test_link.cpp)
target_include_directories(test_link PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(test_link PRIVATE onnxruntime)
set_target_properties(test_link PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${LOADER_ORIGIN}/../third_party/onnxruntime/lib"
)

# 階段二：FP32 基準測試執行檔
add_executable(fp32_benchmark fp32_benchmark.cpp)
target_include_directories(fp32_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(fp32_benchmark PRIVATE onnxruntime)
set_target_properties(fp32_benchmark PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${LOADER_ORIGIN}/../third_party/onnxruntime/lib"
)

# 階段三：INT8 QDQ 量化推論執行檔
add_executable(quantized_infer quantized_infer.cpp)
target_include_directories(quantized_infer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(quantized_infer PRIVATE onnxruntime)
set_target_properties(quantized_infer PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${LOADER_ORIGIN}/../third_party/onnxruntime/lib"
)
