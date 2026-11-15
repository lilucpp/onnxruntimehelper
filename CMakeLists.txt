cmake_minimum_required(VERSION 3.17)
project(OnnxRuntimeHelper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC编译器UTF-8支持
if(MSVC)
    add_compile_options(/utf-8)
endif()

# 可选开启 ASan / UBSan（仅 GCC / Clang）
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(ENABLE_SANITIZERS AND NOT MSVC)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# 设置ONNX Runtime路径（支持通过 -DONNXRUNTIME_ROOT_DIR=... 覆盖）
set(ONNXRUNTIME_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-v1.17.3"
    CACHE PATH "Path to ONNX Runtime root directory")

if(NOT EXISTS "${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime_cxx_api.h")
    message(WARNING "未在 ${ONNXRUNTIME_ROOT_DIR} 找到 ONNX Runtime 头文件，请参考 docs/获取ONNX_Runtime.md 下载配置。")
endif()

# 创建 OnnxRuntimeHelper 核心库
add_library(OnnxRuntimeHelper
    src/OnnxRuntimeHelper.cpp
    include/OnnxRuntimeHelper.h
)

target_include_directories(OnnxRuntimeHelper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_ROOT_DIR}/include
)

target_link_directories(OnnxRuntimeHelper PUBLIC
    ${ONNXRUNTIME_ROOT_DIR}/lib
)

# 链接ONNX Runtime库
if(WIN32)
    target_link_libraries(OnnxRuntimeHelper PUBLIC
        "${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.lib"
    )
else()
    target_link_libraries(OnnxRuntimeHelper PUBLIC onnxruntime)
    set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib")
endif()

# 辅助宏：为Windows目标复制运行时DLL
function(copy_ort_dlls TARGET_NAME)
    if(WIN32)
        if(EXISTS "${ONNXRUNTIME_ROOT_DIR}/bin/onnxruntime.dll")
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_ROOT_DIR}/bin/onnxruntime.dll"
                $<TARGET_FILE_DIR:${TARGET_NAME}>
                COMMENT "复制 onnxruntime.dll 到 ${TARGET_NAME} 输出目录"
            )
        endif()
        if(EXISTS "${ONNXRUNTIME_ROOT_DIR}/bin/onnxruntime_providers_cuda.dll")
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_ROOT_DIR}/bin/onnxruntime_providers_cuda.dll"
                $<TARGET_FILE_DIR:${TARGET_NAME}>
                COMMENT "复制 CUDA provider DLL 到 ${TARGET_NAME} 输出目录"
            )
        endif()
    endif()
endfunction()

# ==================== 示例程序 ====================
add_executable(onnx_helper_test examples/main.cpp)
target_link_libraries(onnx_helper_test PRIVATE OnnxRuntimeHelper)
copy_ort_dlls(onnx_helper_test)

add_executable(check_nlp_models examples/check_nlp_models.cpp)
target_link_libraries(check_nlp_models PRIVATE OnnxRuntimeHelper)
copy_ort_dlls(check_nlp_models)

add_executable(ppocr_and_cls_example examples/ppocr_and_cls_example.cpp)
target_link_libraries(ppocr_and_cls_example PRIVATE OnnxRuntimeHelper)
copy_ort_dlls(ppocr_and_cls_example)

# ==================== 测试程序 ====================
add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic PRIVATE OnnxRuntimeHelper)
copy_ort_dlls(test_basic)

add_executable(unit_tests tests/unit_tests.cpp)
target_link_libraries(unit_tests PRIVATE OnnxRuntimeHelper)
copy_ort_dlls(unit_tests)

# ==================== CTest 注册 ====================
enable_testing()
add_test(NAME test_basic COMMAND test_basic)
add_test(NAME unit_tests COMMAND unit_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# 打印配置信息
message(STATUS "ONNX Runtime 根目录: ${ONNXRUNTIME_ROOT_DIR}")
message(STATUS "包含目录: ${ONNXRUNTIME_ROOT_DIR}/include")
message(STATUS "库目录: ${ONNXRUNTIME_ROOT_DIR}/lib")
