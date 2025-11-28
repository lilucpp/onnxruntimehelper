# 获取 ONNX Runtime 1.17.3

## 问题说明

当前项目需要 ONNX Runtime 1.17.3 的 x64 版本库文件才能正确编译和运行。

## 下载步骤

### Windows x64

1. 访问 ONNX Runtime 官方发布页面：
   <https://github.com/microsoft/onnxruntime/releases/tag/v1.17.3>

2. 下载 Windows x64 版本：
   `onnxruntime-win-x64-1.17.3.zip`

3. 解压下载的文件

4. 将解压后的目录重命名为 `onnxruntime-v1.17.3`

5. 将该目录放置在项目根目录下，最终结构应该是：

   ```
   test_onnxruntime/
   ├── onnxruntime-v1.17.3/
   │   ├── include/
   │   ├── lib/
   │   └── bin/
   ├── OnnxRuntimeHelper.h
   ├── OnnxRuntimeHelper.cpp
   └── ...
   ```

### Linux x64

1. 访问 ONNX Runtime 官方发布页面：
   <https://github.com/microsoft/onnxruntime/releases/tag/v1.17.3>

2. 下载 Linux x64 版本：
   `onnxruntime-linux-x64-1.17.3.tgz`

3. 解压：

   ```bash
   tar -xzf onnxruntime-linux-x64-1.17.3.tgz
   ```

4. 重命名并移动到项目目录：

   ```bash
   mv onnxruntime-linux-x64-1.17.3 onnxruntime-v1.17.3
   ```

## 验证安装

安装完成后，`onnxruntime-v1.17.3` 目录应包含：

```
onnxruntime-v1.17.3/
├── include/
│   ├── onnxruntime_c_api.h
│   ├── onnxruntime_cxx_api.h
│   └── ...
├── lib/
│   ├── onnxruntime.lib (Windows)
│   └── libonnxruntime.so (Linux)
└── bin/ (Windows only)
    └── onnxruntime.dll
```

## 重新构建项目

安装正确的 ONNX Runtime 后，重新构建项目：

```bash
# 清理旧的构建文件
rm -rf build

# 重新配置
cmake -B build -S .

# 编译
cmake --build build --config Release

# 运行
./build/Release/onnx_helper_test  # Windows
./build/onnx_helper_test           # Linux
```

## 其他版本

如果需要使用其他版本的 ONNX Runtime：

1. 访问 <https://github.com/microsoft/onnxruntime/releases>
2. 选择所需版本
3. 下载对应平台的预编译包
4. 按照上述步骤安装

## GPU 支持

如果需要 CUDA 支持，下载包含 GPU 的版本：

- `onnxruntime-win-x64-gpu-1.17.3.zip` (Windows + CUDA)
- `onnxruntime-linux-x64-gpu-1.17.3.tgz` (Linux + CUDA)

注意：GPU 版本需要安装对应的 CUDA 和 cuDNN 库。

## 常见问题

### Q: 找不到 onnxruntime.dll

A: 确保 `onnxruntime-v1.17.3/bin/onnxruntime.dll` 存在。CMakeLists.txt 会自动复制到输出目录。

### Q: 链接错误 "unresolved external symbol OrtGetApiBase"

A: 这通常是因为库文件架构不匹配（x86 vs x64）。请确保下载了正确的 x64 版本。

### Q: 头文件找不到

A: 确保 `onnxruntime-v1.17.3/include/onnxruntime_cxx_api.h` 存在。
