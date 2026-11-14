#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <codecvt>
#include <locale>
#include "OnnxRuntimeHelper.h"

// 读取文本文件的所有行
std::vector<std::string> ReadLines(const std::string& filepath) {
    std::vector<std::string> lines;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filepath << std::endl;
        return lines;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // 移除行尾的回车符
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    
    return lines;
}

// 将中文字符串转换为字符ID序列（简单的字符级编码）
std::vector<int64_t> TextToIds(const std::string& text, int max_length = 512) {
    std::vector<int64_t> ids;
    
    // 简单的UTF-8字符级编码
    for (size_t i = 0; i < text.size() && ids.size() < static_cast<size_t>(max_length); ) {
        unsigned char c = text[i];
        
        if (c < 0x80) {
            // ASCII字符
            ids.push_back(static_cast<int64_t>(c));
            i++;
        } else if (c < 0xE0) {
            // 2字节UTF-8
            if (i + 1 < text.size()) {
                int32_t code = ((c & 0x1F) << 6) | (text[i+1] & 0x3F);
                ids.push_back(static_cast<int64_t>(code));
                i += 2;
            } else {
                i++;
            }
        } else if (c < 0xF0) {
            // 3字节UTF-8（中文主要在这里）
            if (i + 2 < text.size()) {
                int32_t code = ((c & 0x0F) << 12) | 
                              ((text[i+1] & 0x3F) << 6) | 
                              (text[i+2] & 0x3F);
                ids.push_back(static_cast<int64_t>(code % 10000)); // 限制范围
                i += 3;
            } else {
                i++;
            }
        } else {
            // 4字节UTF-8
            i += 4;
        }
    }
    
    // 填充到固定长度
    while (ids.size() < static_cast<size_t>(max_length)) {
        ids.push_back(0);
    }
    
    return ids;
}

// 测试文本分类模型
void TestTextClassifier() {
    std::cout << "\n========== 测试：文本分类模型 ==========" << std::endl;
    
    std::string model_path = "models/text_classifier_rbf.onnx";
    
    try {
        // 加载模型
        OnnxRuntimeHelper helper(model_path);
        std::cout << "✓ 模型加载成功" << std::endl;
        
        // 打印模型信息
        helper.PrintModelInfo();
        
        // 读取测试文本
        auto lines = ReadLines("datas/input.txt");
        std::cout << "\n读取到 " << lines.size() << " 行文本" << std::endl;
        
        // 获取输入输出名称
        auto input_names = helper.GetInputNames();
        auto output_names = helper.GetOutputNames();
        
        // 对每一行文本进行分类
        for (size_t i = 0; i < lines.size(); ++i) {
            std::cout << "\n--- 文本 " << (i+1) << " ---" << std::endl;
            std::cout << "内容: " << lines[i].substr(0, 50) << "..." << std::endl;
            
            try {
                // 这里需要根据实际模型的输入格式来处理
                // 先尝试获取输入形状
                auto input_shape = helper.GetInputShape(0);
                
                std::cout << "输入形状: [";
                for (size_t j = 0; j < input_shape.size(); ++j) {
                    if (j > 0) std::cout << ", ";
                    std::cout << input_shape[j];
                }
                std::cout << "]" << std::endl;
                
                // 注意：这里需要根据实际模型的输入格式来调整
                // 暂时跳过实际推理，只显示模型信息
                
            } catch (const std::exception& e) {
                std::cerr << "处理失败: " << e.what() << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "✗ 错误: " << e.what() << std::endl;
    }
}

// 测试命名实体识别模型
void TestNER() {
    std::cout << "\n========== 测试：命名实体识别模型 ==========" << std::endl;
    
    std::string model_path = "models/ner_bilstm.onnx";
    
    try {
        // 加载模型
        OnnxRuntimeHelper helper(model_path);
        std::cout << "✓ 模型加载成功" << std::endl;
        
        // 打印模型信息
        helper.PrintModelInfo();
        
        // 读取测试文本
        auto lines = ReadLines("datas/input.txt");
        std::cout << "\n读取到 " << lines.size() << " 行文本" << std::endl;
        
        // 获取输入输出名称
        auto input_names = helper.GetInputNames();
        auto output_names = helper.GetOutputNames();
        
        // 对每一行文本进行NER
        for (size_t i = 0; i < lines.size(); ++i) {
            std::cout << "\n--- 文本 " << (i+1) << " ---" << std::endl;
            std::cout << "内容: " << lines[i].substr(0, 50) << "..." << std::endl;
            
            try {
                // 获取输入形状
                auto input_shape = helper.GetInputShape(0);
                
                std::cout << "输入形状: [";
                for (size_t j = 0; j < input_shape.size(); ++j) {
                    if (j > 0) std::cout << ", ";
                    std::cout << input_shape[j];
                }
                std::cout << "]" << std::endl;
                
                // 注意：这里需要根据实际模型的输入格式来调整
                // 暂时跳过实际推理，只显示模型信息
                
            } catch (const std::exception& e) {
                std::cerr << "处理失败: " << e.what() << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "✗ 错误: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  NLP模型信息检查" << std::endl;
    std::cout << "========================================" << std::endl;
    
    TestTextClassifier();
    TestNER();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  检查完成" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}
