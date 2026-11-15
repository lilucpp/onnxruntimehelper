#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "OnnxRuntimeHelper.h"

namespace {

bool FileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

/**
 * @brief 将 RGB 图像数据按指定均值和方差归一化并转换为 NCHW 张量缓冲
 */
std::vector<float> PreprocessRGBToNCHW(
    const std::vector<uint8_t>& rgb,
    int height,
    int width,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& std_dev) {
    const size_t plane_size = static_cast<size_t>(height) * static_cast<size_t>(width);
    std::vector<float> nchw(3 * plane_size, 0.0f);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t pixel_idx = static_cast<size_t>(y) * width + x;
            for (int c = 0; c < 3; ++c) {
                const float val = static_cast<float>(rgb[pixel_idx * 3 + c]) / 255.0f;
                nchw[static_cast<size_t>(c) * plane_size + pixel_idx] =
                    (val - mean[c]) / std_dev[c];
            }
        }
    }
    return nchw;
}

/**
 * @brief 加载 PP-OCR 字符字典（index 0 预留给 CTC blank，末尾追加空格）
 */
std::vector<std::string> LoadPPOCRDict(const std::string& dict_path) {
    std::vector<std::string> dict;
    dict.emplace_back(""); // index 0: CTC blank

    std::ifstream file(dict_path);
    if (!file.is_open()) {
        return dict;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        dict.push_back(line);
    }
    dict.emplace_back(" "); // PP-OCR 字典末尾默认追加空格字符
    return dict;
}

/**
 * @brief CTC 贪心解码（跳过 blank=0 与连续重复索引）
 */
std::pair<std::string, float> CTCGreedyDecode(
    const std::vector<float>& logits,
    int64_t time_steps,
    int64_t num_classes,
    const std::vector<std::string>& dict) {
    std::string text;
    float score_sum = 0.0f;
    int valid_count = 0;
    int64_t prev_idx = 0;

    for (int64_t t = 0; t < time_steps; ++t) {
        const size_t offset = static_cast<size_t>(t * num_classes);
        int64_t best_idx = 0;
        float best_val = logits[offset];

        for (int64_t c = 1; c < num_classes; ++c) {
            const float v = logits[offset + static_cast<size_t>(c)];
            if (v > best_val) {
                best_val = v;
                best_idx = c;
            }
        }

        if (best_idx != 0 && best_idx != prev_idx) {
            if (static_cast<size_t>(best_idx) < dict.size()) {
                text += dict[static_cast<size_t>(best_idx)];
            } else {
                text += "?";
            }
            score_sum += best_val;
            ++valid_count;
        }
        prev_idx = best_idx;
    }

    const float avg_score = (valid_count > 0) ? (score_sum / valid_count) : 0.0f;
    return {text, avg_score};
}

} // namespace

/**
 * @brief 示例 1：图像分类模型推理（如 ResNet18 / MobileNetV3）
 */
void RunImageClassificationExample(const std::string& model_path) {
    std::cout << "\n========== 示例 1：图像分类模型 ==========" << std::endl;
    if (!FileExists(model_path)) {
        std::cout << "未找到分类模型 '" << model_path << "'，跳过此示例。" << std::endl;
        return;
    }

    OnnxRuntimeHelper::SessionConfig config;
    config.providers = {
        OnnxRuntimeHelper::ExecutionProvider::CUDA,
        OnnxRuntimeHelper::ExecutionProvider::CPU
    };
    config.intra_op_num_threads = 4;

    OnnxRuntimeHelper helper(model_path, config);
    helper.PrintModelInfo();

    // 构造 224x224 模拟 RGB 输入并按 ImageNet 参数归一化
    constexpr int kHeight = 224;
    constexpr int kWidth = 224;
    std::vector<uint8_t> dummy_rgb(kHeight * kWidth * 3, 128);
    auto input_data = PreprocessRGBToNCHW(
        dummy_rgb,
        kHeight,
        kWidth,
        {0.485f, 0.456f, 0.406f},
        {0.229f, 0.224f, 0.225f}
    );

    std::vector<Ort::Value> inputs;
    inputs.push_back(helper.CreateTensor(input_data, {1, 3, kHeight, kWidth}));

    auto outputs = helper.Run(
        helper.GetInputNames(),
        inputs,
        helper.GetOutputNames()
    );

    auto logits = OnnxRuntimeHelper::GetTensorData<float>(outputs[0]);
    std::vector<std::pair<size_t, float>> scores;
    scores.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scores.emplace_back(i, logits[i]);
    }

    const size_t top_k = std::min<size_t>(5, scores.size());
    std::partial_sort(
        scores.begin(),
        scores.begin() + top_k,
        scores.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; }
    );

    std::cout << "Top-" << top_k << " 分类结果:" << std::endl;
    for (size_t i = 0; i < top_k; ++i) {
        std::cout << "  " << (i + 1) << ". 类别 " << scores[i].first
                  << " (logit: " << scores[i].second << ")" << std::endl;
    }
}

/**
 * @brief 示例 2：PP-OCRv6 Tiny 文本方向分类 (cls) + 文本识别 (rec)
 */
void RunPPOCRv6TinyExample(
    const std::string& cls_model_path,
    const std::string& rec_model_path,
    const std::string& dict_path) {
    std::cout << "\n========== 示例 2：PP-OCRv6 Tiny 识别 ==========" << std::endl;

    // 1. 可选：文本行方向分类（0° / 180°）
    if (FileExists(cls_model_path)) {
        OnnxRuntimeHelper cls_helper(cls_model_path);
        cls_helper.PrintModelInfo();

        constexpr int kClsH = 48;
        constexpr int kClsW = 192;
        std::vector<uint8_t> cls_rgb(kClsH * kClsW * 3, 200);
        auto cls_input = PreprocessRGBToNCHW(
            cls_rgb,
            kClsH,
            kClsW,
            {0.5f, 0.5f, 0.5f},
            {0.5f, 0.5f, 0.5f}
        );

        std::vector<Ort::Value> cls_inputs;
        cls_inputs.push_back(cls_helper.CreateTensor(cls_input, {1, 3, kClsH, kClsW}));
        auto cls_outputs = cls_helper.Run(
            cls_helper.GetInputNames(),
            cls_inputs,
            cls_helper.GetOutputNames()
        );

        auto cls_probs = OnnxRuntimeHelper::GetTensorData<float>(cls_outputs[0]);
        if (cls_probs.size() >= 2) {
            const int angle = (cls_probs[1] > cls_probs[0]) ? 180 : 0;
            std::cout << "方向分类结果: " << angle << "° (0°=" << cls_probs[0]
                      << ", 180°=" << cls_probs[1] << ")" << std::endl;
        }
    } else {
        std::cout << "未找到方向分类模型 '" << cls_model_path << "'，跳过方向分类步骤。" << std::endl;
    }

    // 2. PP-OCRv6 Tiny 文本识别模型（典型输入 [batch, 3, 48, W]，归一化 mean=0.5, std=0.5）
    if (!FileExists(rec_model_path)) {
        std::cout << "未找到识别模型 '" << rec_model_path << "'，跳过 PP-OCRv6 Tiny 识别示例。" << std::endl;
        return;
    }

    OnnxRuntimeHelper::SessionConfig rec_config;
    rec_config.intra_op_num_threads = 4;
    OnnxRuntimeHelper rec_helper(rec_model_path, rec_config);
    rec_helper.PrintModelInfo();

    constexpr int kRecH = 48;
    constexpr int kRecW = 320;
    std::vector<uint8_t> rec_rgb(kRecH * kRecW * 3, 255);
    auto rec_input = PreprocessRGBToNCHW(
        rec_rgb,
        kRecH,
        kRecW,
        {0.5f, 0.5f, 0.5f},
        {0.5f, 0.5f, 0.5f}
    );

    std::vector<Ort::Value> rec_inputs;
    rec_inputs.push_back(rec_helper.CreateTensor(rec_input, {1, 3, kRecH, kRecW}));

    auto rec_outputs = rec_helper.Run(
        rec_helper.GetInputNames(),
        rec_inputs,
        rec_helper.GetOutputNames()
    );

    // 读取输出形状 [batch, time_steps, num_classes] 并执行 CTC 贪心解码
    auto out_shape = OnnxRuntimeHelper::GetTensorShape(rec_outputs[0]);
    auto out_logits = OnnxRuntimeHelper::GetTensorData<float>(rec_outputs[0]);

    if (out_shape.size() == 3 && out_shape[0] >= 1) {
        const auto dict = LoadPPOCRDict(dict_path);
        auto [text, score] = CTCGreedyDecode(out_logits, out_shape[1], out_shape[2], dict);
        std::cout << "识别文本: \"" << text << "\" (平均置信度: " << score << ")" << std::endl;
    } else {
        std::cout << "识别输出维度: " << out_shape.size() << "，元素数: " << out_logits.size() << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  图像分类 & PP-OCRv6 Tiny 推理示例" << std::endl;
    std::cout << "========================================" << std::endl;

    RunImageClassificationExample("models/resnet18-v1-7.onnx");
    RunPPOCRv6TinyExample(
        "models/ppocr_cls.onnx",
        "models/ppocrv6_tiny_rec.onnx",
        "models/ppocr_keys.txt"
    );

    return 0;
}
