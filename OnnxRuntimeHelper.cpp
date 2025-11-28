#include "OnnxRuntimeHelper.h"
#include <algorithm>
#include <sstream>

// Windows平台字符串转换
#ifdef _WIN32
#include <windows.h>
static std::wstring ToWString(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstr[0], size_needed);
    return wstr;
}
#endif

// ==================== 构造函数 ====================

OnnxRuntimeHelper::OnnxRuntimeHelper(
    const std::string& model_path,
    const std::vector<ExecutionProvider>& providers)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxRuntimeHelper"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    
    // 初始化会话
    InitializeSession(model_path, providers);

    // 缓存输入输出名称
    input_names_ = GetInputNames();
    output_names_ = GetOutputNames();
}

// ==================== 公共方法 ====================

void OnnxRuntimeHelper::SetThreadCount(int intra_op_num_threads, int inter_op_num_threads) {
    if (session_) {
        std::cerr << "警告：会话已创建，线程设置将不会生效。请在构造前设置。" << std::endl;
        return;
    }
    
    session_options_.SetIntraOpNumThreads(intra_op_num_threads);
    if (inter_op_num_threads > 0) {
        session_options_.SetInterOpNumThreads(inter_op_num_threads);
    }
}

void OnnxRuntimeHelper::SetGraphOptimizationLevel(int level) {
    if (session_) {
        std::cerr << "警告：会话已创建，优化级别设置将不会生效。请在构造前设置。" << std::endl;
        return;
    }

    GraphOptimizationLevel opt_level;
    switch (level) {
        case 0:
            opt_level = GraphOptimizationLevel::ORT_DISABLE_ALL;
            break;
        case 1:
            opt_level = GraphOptimizationLevel::ORT_ENABLE_BASIC;
            break;
        case 2:
            opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
            break;
        case 99:
        default:
            opt_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
            break;
    }
    session_options_.SetGraphOptimizationLevel(opt_level);
}

Ort::Value OnnxRuntimeHelper::CreateStringTensor(
    const std::vector<std::string>& strings,
    const std::vector<int64_t>& shape) {
    
    // 计算总大小
    int64_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }

    // 验证数据大小
    if (total_size != static_cast<int64_t>(strings.size())) {
        throw std::runtime_error(
            "字符串数量 (" + std::to_string(strings.size()) + 
            ") 与形状不匹配 (期望 " + std::to_string(total_size) + ")"
        );
    }

    // 准备C风格字符串数组
    std::vector<const char*> c_strings;
    c_strings.reserve(strings.size());
    for (const auto& s : strings) {
        c_strings.push_back(s.c_str());
    }

    // 创建字符串张量
    auto tensor = Ort::Value::CreateTensor(
        allocator_,
        shape.data(),
        shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
    );

    // 填充字符串数据
    tensor.FillStringTensor(c_strings.data(), c_strings.size());

    return tensor;
}

std::vector<Ort::Value> OnnxRuntimeHelper::Run(
    const std::vector<std::string>& input_names,
    std::vector<Ort::Value>& input_tensors,
    const std::vector<std::string>& output_names) {
    
    // 验证输入
    if (input_names.size() != input_tensors.size()) {
        throw std::runtime_error(
            "输入名称数量 (" + std::to_string(input_names.size()) + 
            ") 与输入张量数量 (" + std::to_string(input_tensors.size()) + ") 不匹配"
        );
    }

    // 转换为C风格字符串数组
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

    // 运行推理
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

    size_t count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<std::string> result;
    result.reserve(count);

    // 获取字符串数据
    size_t total_length = tensor.GetStringTensorDataLength();
    std::vector<char> buffer(total_length);
    std::vector<size_t> offsets(count);

    tensor.GetStringTensorContent(buffer.data(), total_length, offsets.data(), count);

    // 转换为字符串向量
    for (size_t i = 0; i < count; ++i) {
        size_t start = offsets[i];
        size_t length = (i < count - 1) ? (offsets[i + 1] - start) : (total_length - start);
        result.emplace_back(buffer.data() + start, length);
    }

    return result;
}

std::vector<int64_t> OnnxRuntimeHelper::GetTensorShape(const Ort::Value& tensor) {
    if (!tensor.IsTensor()) {
        throw std::runtime_error("输出不是张量类型");
    }

    auto type_info = tensor.GetTensorTypeAndShapeInfo();
    return type_info.GetShape();
}

std::vector<std::string> OnnxRuntimeHelper::GetInputNames() const {
    std::vector<std::string> names;
    size_t num_inputs = session_->GetInputCount();
    names.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; ++i) {
        auto name_ptr = session_->GetInputNameAllocated(i, allocator_);
        names.push_back(name_ptr.get());
    }

    return names;
}

std::vector<std::string> OnnxRuntimeHelper::GetOutputNames() const {
    std::vector<std::string> names;
    size_t num_outputs = session_->GetOutputCount();
    names.reserve(num_outputs);

    for (size_t i = 0; i < num_outputs; ++i) {
        auto name_ptr = session_->GetOutputNameAllocated(i, allocator_);
        names.push_back(name_ptr.get());
    }

    return names;
}

std::vector<int64_t> OnnxRuntimeHelper::GetInputShape(size_t index) const {
    if (index >= session_->GetInputCount()) {
        throw std::out_of_range("输入索引超出范围");
    }

    auto type_info = session_->GetInputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

std::vector<int64_t> OnnxRuntimeHelper::GetOutputShape(size_t index) const {
    if (index >= session_->GetOutputCount()) {
        throw std::out_of_range("输出索引超出范围");
    }

    auto type_info = session_->GetOutputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

void OnnxRuntimeHelper::PrintModelInfo() const {
    std::cout << "========== 模型信息 ==========" << std::endl;
    
    // 输入信息
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

    // 输出信息
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

void OnnxRuntimeHelper::InitializeSession(
    const std::string& model_path,
    const std::vector<ExecutionProvider>& providers) {
    
    // 设置默认选项
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 添加执行提供程序
    for (const auto& provider : providers) {
        AppendExecutionProvider(provider);
    }

    // 创建会话
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

void OnnxRuntimeHelper::AppendExecutionProvider(ExecutionProvider provider) {
    try {
        switch (provider) {
            case ExecutionProvider::CPU:
                // CPU是默认提供程序，无需显式添加
                break;

            case ExecutionProvider::CUDA:
                {
                    OrtCUDAProviderOptions cuda_options;
                    cuda_options.device_id = 0;
                    cuda_options.arena_extend_strategy = 0;
                    cuda_options.gpu_mem_limit = SIZE_MAX;
                    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                    cuda_options.do_copy_in_default_stream = 1;
                    session_options_.AppendExecutionProvider_CUDA(cuda_options);
                    std::cout << "已启用 CUDA 执行提供程序" << std::endl;
                }
                break;

            case ExecutionProvider::DirectML:
#ifdef _WIN32
                {
                    session_options_.AppendExecutionProvider("DML");
                    std::cout << "已启用 DirectML 执行提供程序" << std::endl;
                }
#else
                std::cerr << "警告：DirectML仅在Windows平台可用" << std::endl;
#endif
                break;

            case ExecutionProvider::TensorRT:
                {
                    OrtTensorRTProviderOptions trt_options;
                    trt_options.device_id = 0;
                    trt_options.trt_max_workspace_size = 1ULL << 30; // 1GB
                    trt_options.trt_fp16_enable = 1;
                    session_options_.AppendExecutionProvider_TensorRT(trt_options);
                    std::cout << "已启用 TensorRT 执行提供程序" << std::endl;
                }
                break;

            case ExecutionProvider::OpenVINO:
                {
                    session_options_.AppendExecutionProvider("OpenVINO");
                    std::cout << "已启用 OpenVINO 执行提供程序" << std::endl;
                }
                break;

            default:
                std::cerr << "警告：未知的执行提供程序" << std::endl;
                break;
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "警告：添加执行提供程序失败: " << e.what() << std::endl;
        std::cerr << "将回退到CPU执行" << std::endl;
    }
}
