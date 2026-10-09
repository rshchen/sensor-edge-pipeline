# 檔案：cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(SensorEdgeInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 宣告執行檔編譯目標（對齊 cpp/main.cpp 架構）
add_executable(sensor_infer main.cpp)

# 加入專案私有標頭檔目錄 cpp/include
target_include_directories(sensor_infer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# 定義預編譯 ONNX Runtime 本地路徑
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")

# 建立外部匯入庫目標
add_library(onnxruntime SHARED IMPORTED)

# 自動判定作業系統副檔名
if(APPLE)
    set(ORT_LIB_PATH "${ORT_ROOT}/lib/libonnxruntime.dylib")
else()
    set(ORT_LIB_PATH "${ORT_ROOT}/lib/libonnxruntime.so")
endif()

# 配置匯入庫屬性：包含目錄與實體檔案位置
set_target_properties(onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
    IMPORTED_LOCATION "${ORT_LIB_PATH}"
)

# 目標相依性連結
target_link_libraries(sensor_infer PRIVATE onnxruntime)

# 配置執行期動態庫搜尋路徑 (RPATH)
if(APPLE)
    set(LOADER_ORIGIN "@loader_path")
else()
    set(LOADER_ORIGIN "$ORIGIN")
endif()

set_target_properties(sensor_infer PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${LOADER_ORIGIN}/../third_party/onnxruntime/lib"
)

