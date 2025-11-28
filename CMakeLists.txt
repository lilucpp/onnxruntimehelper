cmake_minimum_required(VERSION 3.17)
project(OnnxRuntimeHelperTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC编译器UTF-8支持
if(MSVC)
    add_compile_options(/utf-8)
endif()


# 设置ONNX Runtime路径
set(ONNXRUNTIME_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-v1.17.3")

# 包含目录
include_directories(${ONNXRUNTIME_ROOT_DIR}/include)

# 库目录
link_directories(${ONNXRUNTIME_ROOT_DIR}/lib)

# 创建OnnxRuntimeHelper库
add_library(OnnxRuntimeHelper 
    OnnxRuntimeHelper.cpp
    OnnxRuntimeHelper.h
)

target_include_directories(OnnxRuntimeHelper PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_ROOT_DIR}/include
)

# 链接ONNX Runtime库
if(WIN32)
    # Windows平台使用.lib导入库
    target_link_libraries(OnnxRuntimeHelper PRIVATE 
        "${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.lib"
    )
else()
    target_link_libraries(OnnxRuntimeHelper PRIVATE onnxruntime)
endif()


# 创建测试可执行文件
add_executable(onnx_helper_test main.cpp)

target_link_libraries(onnx_helper_test PRIVATE OnnxRuntimeHelper)

# Windows平台：复制DLL到输出目录
if(WIN32)
    add_custom_command(TARGET onnx_helper_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT_DIR}/bin/onnxruntime.dll"
        $<TARGET_FILE_DIR:onnx_helper_test>
        COMMENT "复制onnxruntime.dll到输出目录"
    )
    
    # 如果有CUDA provider DLL，也复制
    if(EXISTS "${ONNXRUNTIME_ROOT_DIR}/bin/onnxruntime_providers_cuda.dll")
        add_custom_command(TARGET onnx_helper_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT_DIR}/bin/onnxruntime_providers_cuda.dll"
            $<TARGET_FILE_DIR:onnx_helper_test>
            COMMENT "复制CUDA provider DLL到输出目录"
        )
    endif()
endif()

# 打印配置信息
message(STATUS "ONNX Runtime根目录: ${ONNXRUNTIME_ROOT_DIR}")
message(STATUS "包含目录: ${ONNXRUNTIME_ROOT_DIR}/include")
message(STATUS "库目录: ${ONNXRUNTIME_ROOT_DIR}/lib")

# 创建基础测试可执行文件
add_executable(test_basic test_basic.cpp)
target_link_libraries(test_basic PRIVATE OnnxRuntimeHelper)

# 创建单元测试可执行文件
add_executable(unit_tests unit_tests.cpp)
target_link_libraries(unit_tests PRIVATE OnnxRuntimeHelper)

# 创建NLP模型检查程序
add_executable(check_nlp_models check_nlp_models.cpp)
target_link_libraries(check_nlp_models PRIVATE OnnxRuntimeHelper)

# 创建文本分类独立测试程序
add_executable(test_text_classifier test_text_classifier.cpp)
target_link_libraries(test_text_classifier PRIVATE OnnxRuntimeHelper)




