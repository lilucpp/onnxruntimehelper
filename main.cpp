#include <iostream>
#include <vector>
#include <numeric>
#include "OnnxRuntimeHelper.h"

/**
 * 示例1：基本使用 - 浮点数输入输出
 */
void Example1_BasicUsage() {
    std::cout << "\n========== 示例1：基本使用 ==========" << std::endl;
    
    std::string model_path = "model.onnx";
    
    // 检查模型文件是否存在
    FILE* f = fopen(model_path.c_str(), "r");
    if (!f) {
        std::cout << "模型文件 'model.onnx' 不存在，跳过此示例" << std::endl;
        return;
    }
    fclose(f);

    try {
        // 创建Helper实例，使用CPU执行
        OnnxRuntimeHelper helper(model_path);
        
        // 打印模型信息
        helper.PrintModelInfo();
        
        // 创建示例输入数据
        std::vector<int64_t> input_shape = {1, 3, 224, 224};
        size_t input_size = 1 * 3 * 224 * 224;
        std::vector<float> input_data(input_size, 0.5f); // 填充0.5
        
        // 创建输入张量
        auto input_tensor = helper.CreateTensor(input_data, input_shape);
        
        // 获取输入输出名称
        auto input_names = helper.GetInputNames();
        auto output_names = helper.GetOutputNames();
        
        // 准备输入
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(input_tensor));
        
        // 运行推理
        auto outputs = helper.Run(input_names, inputs, output_names);
        
        std::cout << "推理成功！输出数量: " << outputs.size() << std::endl;
        
        // 获取第一个输出的数据
        if (!outputs.empty()) {
            auto output_data = helper.GetTensorData<float>(outputs[0]);
            auto output_shape = helper.GetTensorShape(outputs[0]);
            
            std::cout << "输出形状: [";
            for (size_t i = 0; i < output_shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << output_shape[i];
            }
            std::cout << "]" << std::endl;
            
            std::cout << "输出数据（前10个值）: ";
            for (size_t i = 0; i < std::min(size_t(10), output_data.size()); ++i) {
                std::cout << output_data[i] << " ";
            }
            std::cout << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
    }
}

/**
 * 示例2：使用GPU加速（CUDA）
 */
void Example2_GPUAcceleration() {
    std::cout << "\n========== 示例2：GPU加速 ==========" << std::endl;
    
    std::string model_path = "model.onnx";
    
    FILE* f = fopen(model_path.c_str(), "r");
    if (!f) {
        std::cout << "模型文件 'model.onnx' 不存在，跳过此示例" << std::endl;
        return;
    }
    fclose(f);

    try {
        // 创建Helper实例，优先使用CUDA，回退到CPU
        OnnxRuntimeHelper helper(
            model_path,
            {OnnxRuntimeHelper::ExecutionProvider::CUDA, 
             OnnxRuntimeHelper::ExecutionProvider::CPU}
        );
        
        std::cout << "已创建会话（尝试使用CUDA）" << std::endl;
        helper.PrintModelInfo();
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
    }
}

/**
 * 示例3：多种数据类型
 */
void Example3_MultipleDataTypes() {
    std::cout << "\n========== 示例3：多种数据类型 ==========" << std::endl;
    
    try {
        // 演示不同数据类型的张量创建
        OnnxRuntimeHelper helper("model.onnx");
        
        // Float类型
        std::vector<float> float_data = {1.0f, 2.0f, 3.0f, 4.0f};
        auto float_tensor = helper.CreateTensor(float_data, {2, 2});
        std::cout << "创建Float张量成功" << std::endl;
        
        // Int32类型
        std::vector<int32_t> int_data = {1, 2, 3, 4, 5, 6};
        auto int_tensor = helper.CreateTensor(int_data, {2, 3});
        std::cout << "创建Int32张量成功" << std::endl;
        
        // Int64类型
        std::vector<int64_t> int64_data = {100, 200, 300};
        auto int64_tensor = helper.CreateTensor(int64_data, {3});
        std::cout << "创建Int64张量成功" << std::endl;
        
        // 字符串类型（如果模型支持）
        std::vector<std::string> string_data = {"hello", "world"};
        auto string_tensor = helper.CreateStringTensor(string_data, {2});
        std::cout << "创建String张量成功" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "注意: " << e.what() << std::endl;
    }
}

/**
 * 示例4：批量处理
 */
void Example4_BatchProcessing() {
    std::cout << "\n========== 示例4：批量处理 ==========" << std::endl;
    
    std::string model_path = "model.onnx";
    
    FILE* f = fopen(model_path.c_str(), "r");
    if (!f) {
        std::cout << "模型文件 'model.onnx' 不存在，跳过此示例" << std::endl;
        return;
    }
    fclose(f);

    try {
        OnnxRuntimeHelper helper(model_path);
        
        // 批量大小为4
        int batch_size = 4;
        std::vector<int64_t> input_shape = {batch_size, 3, 224, 224};
        size_t input_size = batch_size * 3 * 224 * 224;
        
        std::vector<float> batch_data(input_size);
        std::iota(batch_data.begin(), batch_data.end(), 0.0f);
        
        auto input_tensor = helper.CreateTensor(batch_data, input_shape);
        
        auto input_names = helper.GetInputNames();
        auto output_names = helper.GetOutputNames();
        
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(input_tensor));
        
        auto outputs = helper.Run(input_names, inputs, output_names);
        
        std::cout << "批量推理成功！批量大小: " << batch_size << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
    }
}

/**
 * 示例5：获取模型元数据
 */
void Example5_ModelMetadata() {
    std::cout << "\n========== 示例5：模型元数据 ==========" << std::endl;
    
    std::string model_path = "model.onnx";
    
    FILE* f = fopen(model_path.c_str(), "r");
    if (!f) {
        std::cout << "模型文件 'model.onnx' 不存在，跳过此示例" << std::endl;
        return;
    }
    fclose(f);

    try {
        OnnxRuntimeHelper helper(model_path);
        
        // 获取输入信息
        auto input_names = helper.GetInputNames();
        std::cout << "输入数量: " << input_names.size() << std::endl;
        
        for (size_t i = 0; i < input_names.size(); ++i) {
            std::cout << "  输入 " << i << ": " << input_names[i];
            
            auto shape = helper.GetInputShape(i);
            std::cout << " - 形状: [";
            for (size_t j = 0; j < shape.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << shape[j];
            }
            std::cout << "]" << std::endl;
        }
        
        // 获取输出信息
        auto output_names = helper.GetOutputNames();
        std::cout << "\n输出数量: " << output_names.size() << std::endl;
        
        for (size_t i = 0; i < output_names.size(); ++i) {
            std::cout << "  输出 " << i << ": " << output_names[i];
            
            auto shape = helper.GetOutputShape(i);
            std::cout << " - 形状: [";
            for (size_t j = 0; j < shape.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << shape[j];
            }
            std::cout << "]" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  ONNX Runtime C++ Helper 示例程序" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 运行所有示例
    Example1_BasicUsage();
    Example2_GPUAcceleration();
    Example3_MultipleDataTypes();
    Example4_BatchProcessing();
    Example5_ModelMetadata();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  所有示例执行完成" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
