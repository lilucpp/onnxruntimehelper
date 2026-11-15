#include "OnnxRuntimeHelper.h"
#include <iostream>

#ifdef _WIN32
#include <windows.h>
static std::wstring ToWString(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.data(), static_cast<int>(str.size()), nullptr, 0);
    if (size_needed <= 0) {
        throw std::runtime_error("UTF-8 路径转换为宽字符失败");
    }
    std::wstring wstr(size_needed, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, str.data(), static_cast<int>(str.size()), &wstr[0], size_needed);
    return wstr;
}
#endif

// ==================== 构造函数 ====================

OnnxRuntimeHelper::OnnxRuntimeHelper(
    const std::string& model_path,
    const SessionConfig& config)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxRuntimeHelper"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

    InitializeSession(model_path, config);

    // 构造时一次性缓存输入输出名称
    const size_t num_inputs = session_->GetInputCount();
    input_names_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name_ptr = session_->GetInputNameAllocated(i, allocator_);
        input_names_.emplace_back(name_ptr.get());
    }

    const size_t num_outputs = session_->GetOutputCount();
    output_names_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name_ptr = session_->GetOutputNameAllocated(i, allocator_);
        output_names_.emplace_back(name_ptr.get());
    }
}

OnnxRuntimeHelper::OnnxRuntimeHelper(
    const std::string& model_path,
    const std::vector<ExecutionProvider>& providers)
    : OnnxRuntimeHelper(
          model_path,
          [&providers]() {
              SessionConfig cfg;
              cfg.providers = providers;
              return cfg;
          }()) {}

// ==================== 公共方法 ====================

Ort::Value OnnxRuntimeHelper::CreateStringTensor(
    const std::vector<std::string>& strings,
    const std::vector<int64_t>& shape) const {
    const size_t expected_count = onnxruntime_helper::detail::ComputeElementCount(shape);
    if (expected_count != strings.size()) {
        throw std::invalid_argument(
            "字符串数量 (" + std::to_string(strings.size()) +
            ") 与形状不匹配 (期望 " + std::to_string(expected_count) + ")"
        );
    }

    std::vector<const char*> c_strings;
    c_strings.reserve(strings.size());
    for (const auto& s : strings) {
        c_strings.push_back(s.c_str());
    }

    auto tensor = Ort::Value::CreateTensor(
        allocator_,
        shape.data(),
        shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
    );

    if (!c_strings.empty()) {
        tensor.FillStringTensor(c_strings.data(), c_strings.size());
    }

    return tensor;
}

std::vector<Ort::Value> OnnxRuntimeHelper::Run(
    const std::vector<std::string>& input_names,
    std::vector<Ort::Value>& input_tensors,
    const std::vector<std::string>& output_names) {
    if (input_names.size() != input_tensors.size()) {
        throw std::invalid_argument(
            "输入名称数量 (" + std::to_string(input_names.size()) +
            ") 与输入张量数量 (" + std::to_string(input_tensors.size()) + ") 不匹配"
        );
    }

    std::vector<const char*> input_names_c;
    input_names_c.reserve(input_names.size());
    for (const auto& name : input_names) {
        input_names_c.push_back(name.c_str());
    }

    std::vector<const char*> output_names_c;
    output_names_c.reserve(output_names.size());
    for (const auto& name : output_names) {
        output_names_c.push_back(name.c_str());
    }

    try {
        return session_->Run(
            Ort::RunOptions{nullptr},
            input_names_c.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_c.data(),
            output_names.size()
        );
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime推理失败: ") + e.what());
    }
}

std::vector<std::string> OnnxRuntimeHelper::GetStringTensorData(const Ort::Value& tensor) {
    if (!tensor.IsTensor()) {
        throw std::runtime_error("输出不是张量类型");
    }

    auto type_info = tensor.GetTensorTypeAndShapeInfo();
    if (type_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        throw std::runtime_error("输出张量不是字符串类型");
    }

    const size_t count = type_info.GetElementCount();
    if (count == 0) {
        return {};
    }

    const size_t total_length = tensor.GetStringTensorDataLength();
    if (total_length == 0) {
        return std::vector<std::string>(count);
    }

    std::vector<char> buffer(total_length);
    std::vector<size_t> offsets(count);
    tensor.GetStringTensorContent(buffer.data(), total_length, offsets.data(), count);

    std::vector<std::string> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t start = offsets[i];
        const size_t end = (i + 1 < count) ? offsets[i + 1] : total_length;
        if (start > end || end > total_length) {
            throw std::runtime_error("字符串张量偏移量无效");
        }
        result.emplace_back(buffer.data() + start, end - start);
    }

    return result;
}

std::vector<int64_t> OnnxRuntimeHelper::GetTensorShape(const Ort::Value& tensor) {
    if (!tensor.IsTensor()) {
        throw std::runtime_error("输出不是张量类型");
    }

    return tensor.GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<int64_t> OnnxRuntimeHelper::GetInputShape(size_t index) const {
    if (index >= session_->GetInputCount()) {
        throw std::out_of_range("输入索引超出范围");
    }

    return session_->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<int64_t> OnnxRuntimeHelper::GetOutputShape(size_t index) const {
    if (index >= session_->GetOutputCount()) {
        throw std::out_of_range("输出索引超出范围");
    }

    return session_->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

void OnnxRuntimeHelper::PrintModelInfo() const {
    std::cout << "========== 模型信息 ==========" << std::endl;

    std::cout << "\n输入 (" << input_names_.size() << "):" << std::endl;
    for (size_t i = 0; i < input_names_.size(); ++i) {
        std::cout << "  [" << i << "] " << input_names_[i];
        try {
            auto shape = GetInputShape(i);
            std::cout << " - 形状: [";
            for (size_t j = 0; j < shape.size(); ++j) {
                if (j > 0) std::cout << ", ";
                if (shape[j] == -1) {
                    std::cout << "动态";
                } else {
                    std::cout << shape[j];
                }
            }
            std::cout << "]";
        } catch (...) {
            std::cout << " - 形状: 未知";
        }
        std::cout << std::endl;
    }

    std::cout << "\n输出 (" << output_names_.size() << "):" << std::endl;
    for (size_t i = 0; i < output_names_.size(); ++i) {
        std::cout << "  [" << i << "] " << output_names_[i];
        try {
            auto shape = GetOutputShape(i);
            std::cout << " - 形状: [";
            for (size_t j = 0; j < shape.size(); ++j) {
                if (j > 0) std::cout << ", ";
                if (shape[j] == -1) {
                    std::cout << "动态";
                } else {
                    std::cout << shape[j];
                }
            }
            std::cout << "]";
        } catch (...) {
            std::cout << " - 形状: 未知";
        }
        std::cout << std::endl;
    }

    std::cout << "=============================" << std::endl;
}

// ==================== 私有方法 ====================

GraphOptimizationLevel OnnxRuntimeHelper::ToOrtOptimizationLevel(int level) {
    switch (level) {
        case 0:
            return GraphOptimizationLevel::ORT_DISABLE_ALL;
        case 1:
            return GraphOptimizationLevel::ORT_ENABLE_BASIC;
        case 2:
            return GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
        case 99:
        default:
            return GraphOptimizationLevel::ORT_ENABLE_ALL;
    }
}

void OnnxRuntimeHelper::InitializeSession(
    const std::string& model_path,
    const SessionConfig& config) {
    if (config.intra_op_num_threads > 0) {
        session_options_.SetIntraOpNumThreads(config.intra_op_num_threads);
    }
    if (config.inter_op_num_threads > 0) {
        session_options_.SetInterOpNumThreads(config.inter_op_num_threads);
    }
    session_options_.SetGraphOptimizationLevel(
        ToOrtOptimizationLevel(config.graph_optimization_level)
    );

    for (const auto& provider : config.providers) {
        AppendExecutionProvider(provider, config);
    }
    if (active_providers_.empty()) {
        active_providers_.push_back(ExecutionProvider::CPU);
    }

    try {
#ifdef _WIN32
        auto model_path_w = ToWString(model_path);
        session_ = std::make_unique<Ort::Session>(env_, model_path_w.c_str(), session_options_);
#else
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(
            std::string("加载模型失败 (") + model_path + "): " + e.what()
        );
    }
}

void OnnxRuntimeHelper::AppendExecutionProvider(
    ExecutionProvider provider,
    const SessionConfig& config) {
    try {
        switch (provider) {
            case ExecutionProvider::CPU:
                active_providers_.push_back(ExecutionProvider::CPU);
                break;

            case ExecutionProvider::CUDA: {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = config.gpu_device_id;
                cuda_options.arena_extend_strategy = 0;
                cuda_options.gpu_mem_limit = SIZE_MAX;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                cuda_options.do_copy_in_default_stream = 1;
                session_options_.AppendExecutionProvider_CUDA(cuda_options);
                active_providers_.push_back(ExecutionProvider::CUDA);
                break;
            }

            case ExecutionProvider::DirectML:
#ifdef _WIN32
                session_options_.AppendExecutionProvider("DML");
                active_providers_.push_back(ExecutionProvider::DirectML);
#else
                if (config.strict_provider) {
                    throw std::runtime_error("DirectML 仅在 Windows 平台可用");
                }
                std::cerr << "警告：DirectML 仅在 Windows 平台可用，将回退到 CPU" << std::endl;
#endif
                break;

            case ExecutionProvider::TensorRT: {
                OrtTensorRTProviderOptions trt_options;
                trt_options.device_id = config.gpu_device_id;
                trt_options.trt_max_workspace_size = 1ULL << 30;
                trt_options.trt_fp16_enable = 1;
                session_options_.AppendExecutionProvider_TensorRT(trt_options);
                active_providers_.push_back(ExecutionProvider::TensorRT);
                break;
            }

            case ExecutionProvider::OpenVINO:
                session_options_.AppendExecutionProvider("OpenVINO");
                active_providers_.push_back(ExecutionProvider::OpenVINO);
                break;

            default:
                if (config.strict_provider) {
                    throw std::invalid_argument("未知的执行提供程序");
                }
                std::cerr << "警告：未知的执行提供程序" << std::endl;
                break;
        }
    } catch (const Ort::Exception& e) {
        if (config.strict_provider) {
            throw std::runtime_error(std::string("添加执行提供程序失败: ") + e.what());
        }
        std::cerr << "警告：添加执行提供程序失败: " << e.what()
                  << "，将回退到 CPU 执行" << std::endl;
    }
}
