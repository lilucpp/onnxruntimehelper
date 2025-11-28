#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <iostream>

/**
 * @brief ONNX Runtime C++ 封装类
 * 
 * 参考Python API设计，提供简洁易用的接口用于：
 * - 加载ONNX模型
 * - 配置执行提供程序（CPU、CUDA、DirectML等）
 * - 设置资源占用（线程数、内存等）
 * - 运行推理
 * - 获取输入输出元数据
 */
class OnnxRuntimeHelper {
public:
    /**
     * @brief 执行提供程序枚举
     */
    enum class ExecutionProvider {
        CPU,        // CPU执行提供程序
        CUDA,       // CUDA执行提供程序（NVIDIA GPU）
        DirectML,   // DirectML执行提供程序（Windows）
        TensorRT,   // TensorRT执行提供程序（NVIDIA）
        OpenVINO    // OpenVINO执行提供程序（Intel）
    };

    /**
     * @brief 构造函数
     * @param model_path ONNX模型文件路径
     * @param providers 执行提供程序列表，按优先级排序（默认仅使用CPU）
     */
    explicit OnnxRuntimeHelper(
        const std::string& model_path,
        const std::vector<ExecutionProvider>& providers = {ExecutionProvider::CPU}
    );

    /**
     * @brief 析构函数
     */
    ~OnnxRuntimeHelper() = default;

    // 禁用拷贝
    OnnxRuntimeHelper(const OnnxRuntimeHelper&) = delete;
    OnnxRuntimeHelper& operator=(const OnnxRuntimeHelper&) = delete;

    /**
     * @brief 设置线程数
     * @param intra_op_num_threads 单个操作内部并行线程数
     * @param inter_op_num_threads 操作间并行线程数（可选）
     */
    void SetThreadCount(int intra_op_num_threads, int inter_op_num_threads = 0);

    /**
     * @brief 设置图优化级别
     * @param level 优化级别：0=禁用, 1=基本, 2=扩展, 99=全部
     */
    void SetGraphOptimizationLevel(int level);

    /**
     * @brief 创建输入张量（数值类型）
     * @tparam T 数据类型（float、int32_t、int64_t等）
     * @param data 输入数据向量
     * @param shape 张量形状
     * @return Ort::Value 张量对象
     */
    template <typename T>
    Ort::Value CreateTensor(const std::vector<T>& data, const std::vector<int64_t>& shape);

    /**
     * @brief 创建字符串张量
     * @param strings 字符串向量
     * @param shape 张量形状
     * @return Ort::Value 字符串张量对象
     */
    Ort::Value CreateStringTensor(const std::vector<std::string>& strings, const std::vector<int64_t>& shape);

    /**
     * @brief 运行推理
     * @param input_names 输入名称列表
     * @param input_tensors 输入张量列表
     * @param output_names 输出名称列表
     * @return 输出张量列表
     */
    std::vector<Ort::Value> Run(
        const std::vector<std::string>& input_names,
        std::vector<Ort::Value>& input_tensors,
        const std::vector<std::string>& output_names
    );

    /**
     * @brief 从张量获取数据（数值类型）
     * @tparam T 数据类型
     * @param tensor 输入张量
     * @return 数据向量
     */
    template <typename T>
    std::vector<T> GetTensorData(const Ort::Value& tensor);

    /**
     * @brief 从张量获取字符串数据
     * @param tensor 输入张量
     * @return 字符串向量
     */
    std::vector<std::string> GetStringTensorData(const Ort::Value& tensor);

    /**
     * @brief 获取张量形状
     * @param tensor 输入张量
     * @return 形状向量
     */
    std::vector<int64_t> GetTensorShape(const Ort::Value& tensor);

    /**
     * @brief 获取输入名称列表
     * @return 输入名称向量
     */
    std::vector<std::string> GetInputNames() const;

    /**
     * @brief 获取输出名称列表
     * @return 输出名称向量
     */
    std::vector<std::string> GetOutputNames() const;

    /**
     * @brief 获取输入形状信息
     * @param index 输入索引
     * @return 形状向量
     */
    std::vector<int64_t> GetInputShape(size_t index) const;

    /**
     * @brief 获取输出形状信息
     * @param index 输出索引
     * @return 形状向量
     */
    std::vector<int64_t> GetOutputShape(size_t index) const;

    /**
     * @brief 打印模型信息
     */
    void PrintModelInfo() const;

private:
    /**
     * @brief 初始化会话
     * @param model_path 模型路径
     * @param providers 执行提供程序列表
     */
    void InitializeSession(const std::string& model_path, const std::vector<ExecutionProvider>& providers);

    /**
     * @brief 添加执行提供程序
     * @param provider 执行提供程序
     */
    void AppendExecutionProvider(ExecutionProvider provider);

    Ort::Env env_;                                      // ONNX Runtime环境
    Ort::SessionOptions session_options_;               // 会话选项
    std::unique_ptr<Ort::Session> session_;             // 推理会话
    Ort::AllocatorWithDefaultOptions allocator_;        // 内存分配器
    Ort::MemoryInfo memory_info_;                       // 内存信息

    std::vector<std::string> input_names_;              // 缓存的输入名称
    std::vector<std::string> output_names_;             // 缓存的输出名称
};

// ==================== 模板实现 ====================

template <typename T>
Ort::Value OnnxRuntimeHelper::CreateTensor(const std::vector<T>& data, const std::vector<int64_t>& shape) {
    // 计算总大小
    int64_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }

    // 验证数据大小
    if (total_size != static_cast<int64_t>(data.size())) {
        throw std::runtime_error(
            "数据大小 (" + std::to_string(data.size()) + 
            ") 与形状不匹配 (期望 " + std::to_string(total_size) + ")"
        );
    }

    // 创建张量
    // 注意：这里使用const_cast是因为ONNX Runtime API的限制
    // 数据在Run调用期间必须保持有效
    return Ort::Value::CreateTensor<T>(
        memory_info_,
        const_cast<T*>(data.data()),
        data.size(),
        shape.data(),
        shape.size()
    );
}

template <typename T>
std::vector<T> OnnxRuntimeHelper::GetTensorData(const Ort::Value& tensor) {
    // 验证是否为张量
    if (!tensor.IsTensor()) {
        throw std::runtime_error("输出不是张量类型");
    }

    // 获取数据指针和元素数量
    const T* raw_data = tensor.GetTensorData<T>();
    size_t count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    // 复制数据到向量
    return std::vector<T>(raw_data, raw_data + count);
}
