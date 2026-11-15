#pragma once

#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace onnxruntime_helper {
namespace detail {

template <typename T>
struct OrtTypeTraits {
    static constexpr bool is_supported = false;
};

#define ORT_HELPER_DEFINE_TYPE_TRAIT(CppType, OrtEnum)                     \
    template <>                                                            \
    struct OrtTypeTraits<CppType> {                                        \
        static constexpr bool is_supported = true;                         \
        static constexpr ONNXTensorElementDataType element_type = OrtEnum; \
    };

ORT_HELPER_DEFINE_TYPE_TRAIT(float, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
ORT_HELPER_DEFINE_TYPE_TRAIT(double, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
ORT_HELPER_DEFINE_TYPE_TRAIT(int8_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
ORT_HELPER_DEFINE_TYPE_TRAIT(uint8_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
ORT_HELPER_DEFINE_TYPE_TRAIT(int16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)
ORT_HELPER_DEFINE_TYPE_TRAIT(uint16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)
ORT_HELPER_DEFINE_TYPE_TRAIT(int32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
ORT_HELPER_DEFINE_TYPE_TRAIT(uint32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)
ORT_HELPER_DEFINE_TYPE_TRAIT(int64_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
ORT_HELPER_DEFINE_TYPE_TRAIT(uint64_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64)
ORT_HELPER_DEFINE_TYPE_TRAIT(bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)

#undef ORT_HELPER_DEFINE_TYPE_TRAIT

/**
 * @brief 计算张量形状对应的元素总数，校验负维度与整数溢出
 */
inline size_t ComputeElementCount(const std::vector<int64_t>& shape) {
    int64_t total_size = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument(
                "张量形状维度不能为负数: " + std::to_string(dim)
            );
        }
        if (dim == 0) {
            total_size = 0;
            continue;
        }
        if (total_size > std::numeric_limits<int64_t>::max() / dim) {
            throw std::overflow_error("张量形状元素总数超出 int64_t 表示范围");
        }
        total_size *= dim;
    }
    return static_cast<size_t>(total_size);
}

} // namespace detail
} // namespace onnxruntime_helper

/**
 * @brief ONNX Runtime C++ 封装类
 *
 * 参考 Python API 设计，提供简洁易用的接口用于：
 * - 加载 ONNX 模型
 * - 配置执行提供程序（CPU、CUDA、DirectML、TensorRT、OpenVINO）
 * - 通过 SessionConfig 设置线程数与图优化级别
 * - 运行推理与安全提取张量数据
 * - 获取输入输出元数据
 */
class OnnxRuntimeHelper {
public:
    /**
     * @brief 执行提供程序枚举
     */
    enum class ExecutionProvider {
        CPU,        // CPU 执行提供程序
        CUDA,       // CUDA 执行提供程序（NVIDIA GPU）
        DirectML,   // DirectML 执行提供程序（Windows）
        TensorRT,   // TensorRT 执行提供程序（NVIDIA）
        OpenVINO    // OpenVINO 执行提供程序（Intel）
    };

    /**
     * @brief 会话初始化配置
     */
    struct SessionConfig {
        std::vector<ExecutionProvider> providers = {ExecutionProvider::CPU};
        int intra_op_num_threads = 1;       // 单算子内部并行线程数
        int inter_op_num_threads = 0;       // 算子间并行线程数（<=0 表示使用默认）
        int graph_optimization_level = 99;  // 0=禁用, 1=基本, 2=扩展, 99=全部
        int gpu_device_id = 0;              // CUDA / TensorRT 设备 ID
        bool strict_provider = false;       // true 时 EP 初始化失败直接抛异常，false 时回退 CPU
    };

    /**
     * @brief 构造函数（使用完整会话配置）
     * @param model_path ONNX 模型文件路径
     * @param config 会话配置
     */
    OnnxRuntimeHelper(
        const std::string& model_path,
        const SessionConfig& config
    );

    /**
     * @brief 便捷构造函数（仅指定执行提供程序列表）
     * @param model_path ONNX 模型文件路径
     * @param providers 执行提供程序列表，按优先级排序（默认仅使用 CPU）
     */
    explicit OnnxRuntimeHelper(
        const std::string& model_path,
        const std::vector<ExecutionProvider>& providers = {ExecutionProvider::CPU}
    );

    ~OnnxRuntimeHelper() = default;

    // 禁用拷贝，允许移动
    OnnxRuntimeHelper(const OnnxRuntimeHelper&) = delete;
    OnnxRuntimeHelper& operator=(const OnnxRuntimeHelper&) = delete;
    OnnxRuntimeHelper(OnnxRuntimeHelper&&) noexcept = default;
    OnnxRuntimeHelper& operator=(OnnxRuntimeHelper&&) noexcept = default;

    /**
     * @brief 创建输入张量（数值类型）
     * @note 零拷贝视图：调用者必须确保 data 在 Run() 完成前保持有效且未重分配
     */
    template <typename T>
    Ort::Value CreateTensor(const std::vector<T>& data, const std::vector<int64_t>& shape) const;

    /**
     * @brief 创建字符串张量
     */
    Ort::Value CreateStringTensor(
        const std::vector<std::string>& strings,
        const std::vector<int64_t>& shape
    ) const;

    /**
     * @brief 运行推理
     */
    std::vector<Ort::Value> Run(
        const std::vector<std::string>& input_names,
        std::vector<Ort::Value>& input_tensors,
        const std::vector<std::string>& output_names
    );

    /**
     * @brief 从张量安全获取数据（校验张量元素类型与模板参数 T 一致）
     */
    template <typename T>
    static std::vector<T> GetTensorData(const Ort::Value& tensor);

    /**
     * @brief 从张量获取字符串数据
     */
    static std::vector<std::string> GetStringTensorData(const Ort::Value& tensor);

    /**
     * @brief 获取张量形状
     */
    static std::vector<int64_t> GetTensorShape(const Ort::Value& tensor);

    /**
     * @brief 获取输入名称列表（返回构造时缓存的元数据）
     */
    const std::vector<std::string>& GetInputNames() const noexcept { return input_names_; }

    /**
     * @brief 获取输出名称列表（返回构造时缓存的元数据）
     */
    const std::vector<std::string>& GetOutputNames() const noexcept { return output_names_; }

    /**
     * @brief 获取实际激活的执行提供程序列表
     */
    const std::vector<ExecutionProvider>& GetActiveProviders() const noexcept { return active_providers_; }

    /**
     * @brief 获取输入形状信息
     */
    std::vector<int64_t> GetInputShape(size_t index) const;

    /**
     * @brief 获取输出形状信息
     */
    std::vector<int64_t> GetOutputShape(size_t index) const;

    /**
     * @brief 打印模型信息
     */
    void PrintModelInfo() const;

private:
    void InitializeSession(const std::string& model_path, const SessionConfig& config);
    void AppendExecutionProvider(ExecutionProvider provider, const SessionConfig& config);
    static GraphOptimizationLevel ToOrtOptimizationLevel(int level);

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    mutable Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<ExecutionProvider> active_providers_;
};

// ==================== 模板实现 ====================

template <typename T>
Ort::Value OnnxRuntimeHelper::CreateTensor(
    const std::vector<T>& data,
    const std::vector<int64_t>& shape) const {
    static_assert(
        onnxruntime_helper::detail::OrtTypeTraits<T>::is_supported,
        "CreateTensor<T>: 不支持的张量数据类型"
    );
    static_assert(
        !std::is_same_v<T, bool>,
        "CreateTensor<bool>: std::vector<bool> 不支持连续内存视图，请使用 uint8_t 代替"
    );

    const size_t expected_count = onnxruntime_helper::detail::ComputeElementCount(shape);
    if (expected_count != data.size()) {
        throw std::invalid_argument(
            "数据大小 (" + std::to_string(data.size()) +
            ") 与形状不匹配 (期望 " + std::to_string(expected_count) + ")"
        );
    }

    // 注意：ONNX Runtime C++ API 要求非 const 指针，此处不会修改数据；
    // 调用者须保证 data 在 Run() 调用期间保持有效。
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
    static_assert(
        onnxruntime_helper::detail::OrtTypeTraits<T>::is_supported,
        "GetTensorData<T>: 不支持的张量数据类型"
    );

    if (!tensor.IsTensor()) {
        throw std::runtime_error("输出不是张量类型");
    }

    auto type_info = tensor.GetTensorTypeAndShapeInfo();
    const auto actual_type = type_info.GetElementType();
    const auto expected_type = onnxruntime_helper::detail::OrtTypeTraits<T>::element_type;
    if (actual_type != expected_type) {
        throw std::runtime_error(
            "张量元素类型不匹配 (实际类型枚举值: " +
            std::to_string(static_cast<int>(actual_type)) +
            ", 期望类型枚举值: " +
            std::to_string(static_cast<int>(expected_type)) + ")"
        );
    }

    const size_t count = type_info.GetElementCount();
    if (count == 0) {
        return {};
    }

    const T* raw_data = tensor.GetTensorData<T>();
    return std::vector<T>(raw_data, raw_data + count);
}
