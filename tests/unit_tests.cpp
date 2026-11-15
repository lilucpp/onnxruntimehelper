#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <map>
#include <iterator>
#include "OnnxRuntimeHelper.h"

// 测试结果统计
struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
    int skipped = 0;
    
    void Pass(const std::string& name) {
        total++;
        passed++;
        std::cout << "  ✓ " << name << std::endl;
    }
    
    void Fail(const std::string& name, const std::string& reason) {
        total++;
        failed++;
        std::cerr << "  ✗ " << name << " - " << reason << std::endl;
    }

    void Skip(const std::string& name, const std::string& reason) {
        total++;
        skipped++;
        std::cout << "  - [SKIP] " << name << " (" << reason << ")" << std::endl;
    }
    
    void PrintSummary() {
        std::cout << "\n========================================" << std::endl;
        std::cout << "  测试总结" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "总计: " << total
                  << " | 通过: " << passed
                  << " | 跳过: " << skipped
                  << " | 失败: " << failed << std::endl;
        if (failed == 0) {
            std::cout << "✓ 所有已执行测试通过！" << std::endl;
        } else {
            std::cout << "✗ 有 " << failed << " 个测试失败" << std::endl;
        }
        std::cout << "========================================\n" << std::endl;
    }
};

// 检查文件是否存在
bool FileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// 读取文本文件的所有行
std::vector<std::string> ReadLines(const std::string& filepath) {
    std::vector<std::string> lines;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
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

// 简单的JSON解析 - 读取标签映射
std::map<int, std::string> LoadLabelMap(const std::string& filepath) {
    std::map<int, std::string> label_map;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        return label_map;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    
    // 简单的JSON解析（仅适用于这个特定格式）
    size_t pos = 0;
    while ((pos = content.find("\"", pos)) != std::string::npos) {
        size_t start = pos + 1;
        size_t end = content.find("\"", start);
        if (end == std::string::npos) break;
        
        std::string label = content.substr(start, end - start);
        
        // 查找对应的数字
        pos = content.find(":", end);
        if (pos == std::string::npos) break;
        
        pos++;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        
        size_t num_end = pos;
        while (num_end < content.size() && (content[num_end] >= '0' && content[num_end] <= '9')) {
            num_end++;
        }
        
        if (num_end > pos) {
            int value = std::stoi(content.substr(pos, num_end - pos));
            label_map[value] = label;
        }
        
        pos = num_end;
    }
    
    return label_map;
}


// 简单的文本转ID（用于NER模型）
std::vector<int64_t> TextToIds(const std::string& text, int max_length = 128) {
    std::vector<int64_t> ids;
    
    // 简单的字符级编码，限制在词汇表范围内
    const int vocab_size = 1077;  // NER模型的词汇表大小
    
    for (size_t i = 0; i < text.size() && ids.size() < static_cast<size_t>(max_length); ) {
        unsigned char c = text[i];
        
        if (c < 0x80) {
            // ASCII
            ids.push_back(static_cast<int64_t>(c % vocab_size));
            i++;
        } else if (c < 0xE0) {
            // 2字节UTF-8
            if (i + 1 < text.size()) {
                int32_t code = ((c & 0x1F) << 6) | (text[i+1] & 0x3F);
                ids.push_back(static_cast<int64_t>(code % vocab_size));
                i += 2;
            } else {
                i++;
            }
        } else if (c < 0xF0) {
            // 3字节UTF-8（中文）
            if (i + 2 < text.size()) {
                int32_t code = ((c & 0x0F) << 12) | 
                              ((text[i+1] & 0x3F) << 6) | 
                              (text[i+2] & 0x3F);
                ids.push_back(static_cast<int64_t>(code % vocab_size));
                i += 3;
            } else {
                i++;
            }
        } else {
            i += 4;
        }
    }
    
    // 填充到固定长度
    while (ids.size() < static_cast<size_t>(max_length)) {
        ids.push_back(0);
    }
    
    return ids;
}



// 测试1：MNIST手写数字识别模型
void TestMNIST(TestStats& stats) {
    std::cout << "\n========== 测试1：MNIST手写数字识别 ==========" << std::endl;
    
    std::string model_path = "models/mnist-12.onnx";
    
    if (!FileExists(model_path)) {
        stats.Skip("MNIST测试", "模型文件不存在: " + model_path);
        return;
    }
    
    try {
        // 创建Helper实例
        OnnxRuntimeHelper helper(model_path);
        stats.Pass("MNIST模型加载");
        
        // 打印模型信息
        helper.PrintModelInfo();
        
        // 获取输入输出信息
        auto input_names = helper.GetInputNames();
        auto output_names = helper.GetOutputNames();
        
        if (input_names.size() != 1) {
            stats.Fail("MNIST输入验证", "期望1个输入，实际" + std::to_string(input_names.size()) + "个");
            return;
        }
        stats.Pass("MNIST输入验证");
        
        if (output_names.size() != 1) {
            stats.Fail("MNIST输出验证", "期望1个输出，实际" + std::to_string(output_names.size()) + "个");
            return;
        }
        stats.Pass("MNIST输出验证");
        
        // 创建测试输入 (1, 1, 28, 28) - 一个手写数字图像
        std::vector<int64_t> input_shape = {1, 1, 28, 28};
        std::vector<float> input_data(1 * 1 * 28 * 28, 0.0f);
        
        // 创建一个简单的"1"的图案（中间一列）
        for (int i = 0; i < 28; ++i) {
            input_data[i * 28 + 14] = 1.0f;  // 中间列
        }
        
        auto input_tensor = helper.CreateTensor(input_data, input_shape);
        stats.Pass("MNIST输入张量创建");
        
        // 运行推理
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(input_tensor));
        
        auto outputs = helper.Run(input_names, inputs, output_names);
        stats.Pass("MNIST推理执行");
        
        // 获取输出
        auto output_data = helper.GetTensorData<float>(outputs[0]);
        
        if (output_data.size() != 10) {
            stats.Fail("MNIST输出大小", "期望10个类别，实际" + std::to_string(output_data.size()) + "个");
            return;
        }
        stats.Pass("MNIST输出大小验证");
        
        // 找到预测的数字
        auto max_iter = std::max_element(output_data.begin(), output_data.end());
        int predicted_digit = std::distance(output_data.begin(), max_iter);
        
        std::cout << "  预测数字: " << predicted_digit << " (置信度: " << *max_iter << ")" << std::endl;
        stats.Pass("MNIST预测结果获取");
        
    } catch (const std::exception& e) {
        stats.Fail("MNIST测试", std::string("异常: ") + e.what());
    }
}

// 测试2：ResNet18图像分类模型
void TestResNet18(TestStats& stats) {
    std::cout << "\n========== 测试2：ResNet18图像分类 ==========" << std::endl;
    
    std::string model_path = "models/resnet18-v1-7.onnx";
    
    if (!FileExists(model_path)) {
        stats.Skip("ResNet18测试", "模型文件不存在: " + model_path);
        return;
    }
    
    try {
        // 创建Helper实例
        OnnxRuntimeHelper helper(model_path);
        stats.Pass("ResNet18模型加载");
        
        // 打印模型信息
        helper.PrintModelInfo();
        
        // 获取输入输出信息
        auto input_names = helper.GetInputNames();
        auto output_names = helper.GetOutputNames();
        
        stats.Pass("ResNet18元数据获取");
        
        // 创建测试输入 (1, 3, 224, 224) - RGB图像
        std::vector<int64_t> input_shape = {1, 3, 224, 224};
        std::vector<float> input_data(1 * 3 * 224 * 224);
        
        // 填充随机数据（模拟图像）
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = static_cast<float>(i % 256) / 255.0f;
        }
        
        auto input_tensor = helper.CreateTensor(input_data, input_shape);
        stats.Pass("ResNet18输入张量创建");
        
        // 运行推理
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(input_tensor));
        
        auto outputs = helper.Run(input_names, inputs, output_names);
        stats.Pass("ResNet18推理执行");
        
        // 获取输出
        auto output_data = helper.GetTensorData<float>(outputs[0]);
        
        if (output_data.size() != 1000) {
            stats.Fail("ResNet18输出大小", "期望1000个类别，实际" + std::to_string(output_data.size()) + "个");
            return;
        }
        stats.Pass("ResNet18输出大小验证");
        
        // 找到Top-5预测
        std::vector<std::pair<int, float>> predictions;
        for (size_t i = 0; i < output_data.size(); ++i) {
            predictions.push_back({static_cast<int>(i), output_data[i]});
        }
        
        std::partial_sort(predictions.begin(), predictions.begin() + 5, predictions.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::cout << "  Top-5预测:" << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << "    " << (i+1) << ". 类别 " << predictions[i].first 
                     << " (得分: " << predictions[i].second << ")" << std::endl;
        }
        
        stats.Pass("ResNet18 Top-5预测");
        
    } catch (const std::exception& e) {
        stats.Fail("ResNet18测试", std::string("异常: ") + e.what());
    }
}

// 测试3：多种数据类型
void TestDataTypes(TestStats& stats) {
    std::cout << "\n========== 测试3：多种数据类型支持 ==========" << std::endl;
    
    try {
        // 测试Float类型
        std::vector<float> float_data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> shape = {2, 2};
        
        // 注意：这里我们不需要实际的模型，只测试张量创建
        // 但CreateTensor需要memory_info，所以我们需要一个Helper实例
        if (FileExists("models/mnist-12.onnx")) {
            OnnxRuntimeHelper helper("models/mnist-12.onnx");
            
            auto float_tensor = helper.CreateTensor(float_data, shape);
            stats.Pass("Float类型张量创建");
            
            // 测试Int32类型
            std::vector<int32_t> int_data = {1, 2, 3, 4};
            auto int_tensor = helper.CreateTensor(int_data, shape);
            stats.Pass("Int32类型张量创建");
            
            // 测试Int64类型
            std::vector<int64_t> int64_data = {100, 200, 300, 400};
            auto int64_tensor = helper.CreateTensor(int64_data, shape);
            stats.Pass("Int64类型张量创建");
            
            // 测试Double类型
            std::vector<double> double_data = {1.1, 2.2, 3.3, 4.4};
            auto double_tensor = helper.CreateTensor(double_data, shape);
            stats.Pass("Double类型张量创建");
            
        } else {
            stats.Skip("数据类型测试", "需要MNIST模型文件");
        }
        
    } catch (const std::exception& e) {
        stats.Fail("数据类型测试", std::string("异常: ") + e.what());
    }
}

// 测试4：批量处理
void TestBatchProcessing(TestStats& stats) {
    std::cout << "\n========== 测试4：批量处理 ==========" << std::endl;
    
    // 使用ResNet18，因为它支持动态批量大小
    std::string model_path = "models/resnet18-v1-7.onnx";
    
    if (!FileExists(model_path)) {
        stats.Skip("批量处理测试", "ResNet18模型不存在");
        return;
    }
    
    try {
        OnnxRuntimeHelper helper(model_path);
        
        // 批量大小为4
        int batch_size = 4;
        std::vector<int64_t> input_shape = {batch_size, 3, 224, 224};
        std::vector<float> batch_data(batch_size * 3 * 224 * 224, 0.5f);
        
        auto input_tensor = helper.CreateTensor(batch_data, input_shape);
        stats.Pass("批量输入张量创建");
        
        auto input_names = helper.GetInputNames();
        auto output_names = helper.GetOutputNames();
        
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(input_tensor));
        
        auto outputs = helper.Run(input_names, inputs, output_names);
        stats.Pass("批量推理执行");
        
        auto output_data = helper.GetTensorData<float>(outputs[0]);
        
        // 输出应该是 (batch_size, 1000)
        if (output_data.size() != static_cast<size_t>(batch_size * 1000)) {
            stats.Fail("批量输出验证", "输出大小不正确");
            return;
        }
        stats.Pass("批量输出验证");
        
        std::cout << "  批量大小: " << batch_size << std::endl;
        std::cout << "  输出形状: [" << batch_size << ", 1000]" << std::endl;
        
    } catch (const std::exception& e) {
        stats.Fail("批量处理测试", std::string("异常: ") + e.what());
    }
}


// 测试5：错误处理
void TestErrorHandling(TestStats& stats) {
    std::cout << "\n========== 测试5：错误处理 ==========" << std::endl;
    
    // 测试不存在的模型文件
    try {
        OnnxRuntimeHelper helper("nonexistent_model.onnx");
        stats.Fail("不存在模型错误处理", "应该抛出异常");
    } catch (const std::exception&) {
        stats.Pass("不存在模型错误处理");
    }

    // 测试负维度形状校验（无需模型文件）
    try {
        onnxruntime_helper::detail::ComputeElementCount({-1, 4});
        stats.Fail("负维度形状校验", "应该抛出 invalid_argument 异常");
    } catch (const std::invalid_argument&) {
        stats.Pass("负维度形状校验");
    }

    // 测试零维度形状校验（无需模型文件）
    try {
        size_t zero_count = onnxruntime_helper::detail::ComputeElementCount({2, 0, 5});
        if (zero_count == 0) {
            stats.Pass("零维度形状校验");
        } else {
            stats.Fail("零维度形状校验", "期望元素数为 0");
        }
    } catch (const std::exception& e) {
        stats.Fail("零维度形状校验", e.what());
    }

    // 测试形状乘积溢出校验（无需模型文件）
    try {
        onnxruntime_helper::detail::ComputeElementCount({10000000000LL, 10000000000LL});
        stats.Fail("形状溢出校验", "应该抛出 overflow_error 异常");
    } catch (const std::overflow_error&) {
        stats.Pass("形状溢出校验");
    }

    // 测试形状不匹配与类型不匹配（使用 MNIST 模型）
    if (FileExists("models/mnist-12.onnx")) {
        try {
            OnnxRuntimeHelper helper("models/mnist-12.onnx");

            // 错误的数据大小
            std::vector<float> wrong_data = {1.0f, 2.0f};
            std::vector<int64_t> shape = {2, 2};
            auto tensor = helper.CreateTensor(wrong_data, shape);
            stats.Fail("形状不匹配错误处理", "应该抛出异常");
        } catch (const std::exception&) {
            stats.Pass("形状不匹配错误处理");
        }

        // 测试 GetTensorData<T> 类型不匹配校验
        try {
            OnnxRuntimeHelper helper("models/mnist-12.onnx");
            std::vector<float> float_data = {1.0f, 2.0f};
            auto float_tensor = helper.CreateTensor(float_data, {2});
            (void)OnnxRuntimeHelper::GetTensorData<int32_t>(float_tensor);
            stats.Fail("张量类型不匹配校验", "应该抛出异常");
        } catch (const std::runtime_error&) {
            stats.Pass("张量类型不匹配校验");
        }

        // 测试字符串张量往返（包含空字符串与中文）
        try {
            OnnxRuntimeHelper helper("models/mnist-12.onnx");
            std::vector<std::string> texts = {"hello", "", "中文测试"};
            auto str_tensor = helper.CreateStringTensor(texts, {3});
            auto decoded = OnnxRuntimeHelper::GetStringTensorData(str_tensor);
            if (decoded == texts) {
                stats.Pass("字符串张量往返一致性");
            } else {
                stats.Fail("字符串张量往返一致性", "解码结果与输入不一致");
            }

            std::vector<std::string> empty_texts = {"", ""};
            auto empty_str_tensor = helper.CreateStringTensor(empty_texts, {2});
            auto decoded_empty = OnnxRuntimeHelper::GetStringTensorData(empty_str_tensor);
            if (decoded_empty == empty_texts) {
                stats.Pass("全空字符串张量往返");
            } else {
                stats.Fail("全空字符串张量往返", "全空字符串解码结果不一致");
            }
        } catch (const std::exception& e) {
            stats.Fail("字符串张量往返测试", e.what());
        }

        // 测试 SessionConfig 与非严格回退状态
        try {
            OnnxRuntimeHelper::SessionConfig cfg;
            cfg.providers = {OnnxRuntimeHelper::ExecutionProvider::CPU};
            cfg.intra_op_num_threads = 2;
            cfg.graph_optimization_level = 1;
            OnnxRuntimeHelper helper("models/mnist-12.onnx", cfg);
            if (!helper.GetActiveProviders().empty() &&
                helper.GetActiveProviders()[0] == OnnxRuntimeHelper::ExecutionProvider::CPU) {
                stats.Pass("SessionConfig 配置与 ActiveProviders 查询");
            } else {
                stats.Fail("SessionConfig 配置与 ActiveProviders 查询", "ActiveProviders 为空或不正确");
            }
        } catch (const std::exception& e) {
            stats.Fail("SessionConfig 配置测试", e.what());
        }
    }
}

// 测试6：文本分类模型
void TestTextClassifier(TestStats& stats) {
    std::cout << "\n========== 测试6：文本分类模型 ==========" << std::endl;
    
    std::string model_path = "models/text_classifier_rbf.onnx";
    
    if (!FileExists(model_path)) {
        stats.Skip("文本分类测试", "文本分类模型不存在");
        return;
    }
    
    try {
        // 加载模型
        OnnxRuntimeHelper helper(model_path);
        stats.Pass("文本分类模型加载");
        
        // 读取标签映射
        auto label_map = LoadLabelMap("datas/label_encoder_map.json");
        if (!label_map.empty()) {
            std::cout << "  ✓ 加载标签映射: " << label_map.size() << " 个类别" << std::endl;
        }
        
        // 读取测试文本
        auto lines = ReadLines("datas/input.txt");
        
        if (lines.empty()) {
            stats.Fail("读取测试文本", "input.txt为空或不存在");
            return;
        }
        stats.Pass("读取测试文本");
        
        std::cout << "  读取到 " << lines.size() << " 行文本\n" << std::endl;
        
        // 获取输入输出名称
        auto input_names = helper.GetInputNames();
        auto output_names = helper.GetOutputNames();
        
        // 对所有文本进行分类
        for (size_t idx = 0; idx < lines.size(); ++idx) {
            std::string test_text = lines[idx];
            std::cout << "--- 文本 " << (idx + 1) << " ---" << std::endl;
            std::cout << "内容: " << test_text.substr(0, 40) << "..." << std::endl;
            
            // 创建字符串张量
            std::vector<std::string> texts = {test_text};
            auto input_tensor = helper.CreateStringTensor(texts, {1});
            
            // 运行推理
            std::vector<Ort::Value> inputs;
            inputs.push_back(std::move(input_tensor));
            
            auto outputs = helper.Run(input_names, inputs, output_names);
            
            // 获取分类结果
            if (outputs.size() >= 2) {
                // 输出0: label, 输出1: probabilities
                auto probs = helper.GetTensorData<float>(outputs[1]);
                
                // 找到最大概率的类别
                auto max_iter = std::max_element(probs.begin(), probs.end());
                int predicted_class = std::distance(probs.begin(), max_iter);
                
                // 显示结果
                std::cout << "预测类别: " << predicted_class;
                if (label_map.find(predicted_class) != label_map.end()) {
                    std::cout << " (" << label_map[predicted_class] << ")";
                }
                std::cout << " - 置信度: " << *max_iter << std::endl;
                
                // 显示所有类别的概率
                std::cout << "各类别概率: ";
                for (size_t i = 0; i < probs.size(); ++i) {
                    if (label_map.find(i) != label_map.end()) {
                        std::cout << label_map[i] << ":" << probs[i] << " ";
                    }
                }
                std::cout << "\n" << std::endl;
            }
        }
        
        stats.Pass("文本分类推理执行");
        stats.Pass("文本分类结果获取");
        
    } catch (const std::exception& e) {
        stats.Fail("文本分类测试", std::string("异常: ") + e.what());
    }
}


// 测试7：命名实体识别模型
void TestNER(TestStats& stats) {
    std::cout << "\n========== 测试7：命名实体识别(NER)模型 ==========" << std::endl;
    
    std::string model_path = "models/ner_bilstm.onnx";
    
    if (!FileExists(model_path)) {
        stats.Skip("NER测试", "NER模型不存在");
        return;
    }
    
    try {
        // 加载模型
        OnnxRuntimeHelper helper(model_path);
        stats.Pass("NER模型加载");
        
        // 读取测试文本
        auto lines = ReadLines("datas/input.txt");
        
        if (lines.empty()) {
            stats.Fail("读取测试文本", "input.txt为空或不存在");
            return;
        }
        stats.Pass("读取测试文本");
        
        std::cout << "  读取到 " << lines.size() << " 行文本" << std::endl;
        
        // 获取输入输出名称
        auto input_names = helper.GetInputNames();
        auto output_names = helper.GetOutputNames();
        
        // 对第一行文本进行NER测试
        std::string test_text = lines[0];
        std::cout << "  测试文本: " << test_text.substr(0, 30) << "..." << std::endl;
        
        // 将文本转换为ID序列
        int seq_len = 128;
        auto input_ids = TextToIds(test_text, seq_len);
        
        // 创建mask（全1表示有效token）
        std::vector<int64_t> mask(seq_len, 1);
        
        // 创建输入张量
        auto ids_tensor = helper.CreateTensor(input_ids, {1, seq_len});
        auto mask_tensor = helper.CreateTensor(mask, {1, seq_len});
        stats.Pass("创建NER输入张量");
        
        // 运行推理
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(ids_tensor));
        inputs.push_back(std::move(mask_tensor));
        
        auto outputs = helper.Run(input_names, inputs, output_names);
        stats.Pass("NER推理执行");
        
        // 获取NER结果
        auto logits = helper.GetTensorData<float>(outputs[0]);
        
        // logits形状应该是 [1, seq_len, 7]
        int num_labels = 7;
        std::cout << "  输出logits大小: " << logits.size() << std::endl;
        std::cout << "  期望大小: " << (seq_len * num_labels) << " (seq_len=" << seq_len << ", num_labels=" << num_labels << ")" << std::endl;
        
        // 统计预测的实体标签
        std::vector<int> label_counts(num_labels, 0);
        for (int i = 0; i < seq_len; ++i) {
            // 找到每个位置的最大logit对应的标签
            float max_logit = logits[i * num_labels];
            int max_label = 0;
            for (int j = 1; j < num_labels; ++j) {
                if (logits[i * num_labels + j] > max_logit) {
                    max_logit = logits[i * num_labels + j];
                    max_label = j;
                }
            }
            label_counts[max_label]++;
        }
        
        std::cout << "  实体标签分布: ";
        for (int i = 0; i < num_labels; ++i) {
            std::cout << "L" << i << ":" << label_counts[i] << " ";
        }
        std::cout << std::endl;
        
        stats.Pass("NER结果获取");
        
    } catch (const std::exception& e) {
        stats.Fail("NER测试", std::string("异常: ") + e.what());
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  ONNX Runtime C++ Helper 单元测试" << std::endl;
    std::cout << "========================================" << std::endl;
    
    TestStats stats;
    
    // 运行所有测试
    TestMNIST(stats);
    TestResNet18(stats);
    TestDataTypes(stats);
    TestBatchProcessing(stats);
    TestErrorHandling(stats);
    TestTextClassifier(stats);  // 新增
    TestNER(stats);              // 新增
    
    // 打印总结
    stats.PrintSummary();
    
    return (stats.failed == 0) ? 0 : 1;
}

