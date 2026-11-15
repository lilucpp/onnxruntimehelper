#!/usr/bin/env python3
"""
一键拉取 ONNX Runtime 1.17.3、测试/示例模型并执行编译与测试验证脚本。
用法:
    python3 scripts/setup_and_build.py
    python3 scripts/setup_and_build.py --sanitizers --force
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ORT_VERSION = "1.17.3"
ORT_DIR = ROOT_DIR / f"onnxruntime-v{ORT_VERSION}"
MODELS_DIR = ROOT_DIR / "models"

MODEL_SOURCES = {
    "mnist-12.onnx": "https://huggingface.co/onnxmodelzoo/mnist-12/resolve/main/mnist-12.onnx",
    "resnet18-v1-7.onnx": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx",
    "ppocrv6_tiny_rec.onnx": "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/english/rec.onnx",
    "ppocr_keys.txt": "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/english/dict.txt",
}


def download_file(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"[跳过] 已存在: {dest.relative_to(ROOT_DIR)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[下载] {url} -> {dest.relative_to(ROOT_DIR)}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def setup_onnxruntime(force: bool = False) -> None:
    header_check = ORT_DIR / "include" / "onnxruntime_cxx_api.h"
    if header_check.exists() and not force:
        print(f"[跳过] ONNX Runtime 已就绪: {ORT_DIR.relative_to(ROOT_DIR)}")
        return

    sys_name = platform.system().lower()
    if sys_name == "linux":
        pkg_name = f"onnxruntime-linux-x64-{ORT_VERSION}"
        ext = ".tgz"
    elif sys_name == "windows":
        pkg_name = f"onnxruntime-win-x64-{ORT_VERSION}"
        ext = ".zip"
    else:
        raise RuntimeError(f"暂不支持自动拉取平台: {sys_name}")

    url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{pkg_name}{ext}"
    archive_path = ROOT_DIR / f"{pkg_name}{ext}"

    download_file(url, archive_path, force=True)
    if ORT_DIR.exists():
        shutil.rmtree(ORT_DIR)

    print(f"[解压] {archive_path.name} ...")
    if ext == ".tgz":
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(ROOT_DIR)
    else:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(ROOT_DIR)

    extracted = ROOT_DIR / pkg_name
    extracted.rename(ORT_DIR)
    archive_path.unlink(missing_ok=True)
    print(f"[完成] ONNX Runtime 已安装到: {ORT_DIR}")


def setup_models(force: bool = False) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_SOURCES.items():
        download_file(url, MODELS_DIR / filename, force=force)


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"[执行] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="一键拉取依赖、模型并编译测试 OnnxRuntimeHelper")
    parser.add_argument("--build-dir", default="build", help="CMake 构建目录 (默认: build)")
    parser.add_argument("--sanitizers", action="store_true", help="启用 ASan / UBSan")
    parser.add_argument("--skip-ort", action="store_true", help="跳过拉取 ONNX Runtime")
    parser.add_argument("--skip-models", action="store_true", help="跳过拉取模型文件")
    parser.add_argument("--force", action="store_true", help="强制重新下载已有依赖和模型")
    args = parser.parse_args()

    if not args.skip_ort:
        setup_onnxruntime(force=args.force)

    if not args.skip_models:
        setup_models(force=args.force)

    cmake_args = ["cmake", "-B", args.build_dir, "-S", "."]
    if args.sanitizers:
        cmake_args.append("-DENABLE_SANITIZERS=ON")
    run_cmd(cmake_args)

    build_cmd = ["cmake", "--build", args.build_dir, "-j"]
    if platform.system().lower() == "windows":
        build_cmd = ["cmake", "--build", args.build_dir, "--config", "Release"]
    run_cmd(build_cmd)

    env = os.environ.copy()
    if args.sanitizers:
        # libonnxruntime.so 内部一次性全局 calloc 在 LeakSanitizer 下会误报，禁用泄漏检测保留 ASan/UBSan
        env["ASAN_OPTIONS"] = "detect_leaks=0"

    ctest_cmd = ["ctest", "--test-dir", args.build_dir, "--output-on-failure"]
    if platform.system().lower() == "windows":
        ctest_cmd.extend(["-C", "Release"])
    run_cmd(ctest_cmd, env=env)

    exe_path = ROOT_DIR / args.build_dir / (
        "Release/ppocr_and_cls_example.exe"
        if platform.system().lower() == "windows"
        else "ppocr_and_cls_example"
    )
    run_cmd([str(exe_path)], env=env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
