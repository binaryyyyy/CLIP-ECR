"""
GPU问题检测和修复脚本
"""

import os
import sys
import subprocess

def run_command(cmd):
    """运行命令并返回输出"""
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        return f"错误: {e.output.decode()}"

def check_system():
    """检查系统环境"""
    print("检查系统环境...")
    print(f"Python版本: {sys.version}")
    print(f"系统平台: {sys.platform}")
    
    # 检查NVIDIA驱动
    print("\n检查NVIDIA驱动...")
    nvidia_smi = run_command("nvidia-smi")
    if "NVIDIA-SMI" in nvidia_smi:
        print("✅ NVIDIA驱动已安装")
        print(nvidia_smi.split("\n")[0:6])
    else:
        print("❌ NVIDIA驱动未安装或无法访问")
        print("请安装或更新NVIDIA驱动: https://www.nvidia.com/Download/index.aspx")
    
    # 检查CUDA
    print("\n检查CUDA工具包...")
    if "CUDA Version" in nvidia_smi:
        import re
        match = re.search(r"CUDA Version: (\d+\.\d+)", nvidia_smi)
        if match:
            cuda_version = match.group(1)
            print(f"✅ CUDA工具包已安装 (版本: {cuda_version})")
        else:
            print("❌ 无法确定CUDA版本")
    else:
        print("❌ CUDA工具包未安装或无法访问")
        print("请安装CUDA工具包: https://developer.nvidia.com/cuda-downloads")

def check_pytorch():
    """检查PyTorch安装情况"""
    print("\n检查PyTorch安装情况...")
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✅ PyTorch可以使用CUDA")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ PyTorch无法使用CUDA")
            print("可能的原因:")
            print("1. 安装了CPU版本的PyTorch")
            print("2. CUDA版本与PyTorch不兼容")
            print("3. 环境变量配置不正确")
            
            # 建议解决方案
            print("\n建议解决方案:")
            print("安装与当前CUDA版本兼容的PyTorch:")
            print("python scripts/install_gpu_pytorch.py")
    except ImportError:
        print("❌ 未安装PyTorch")
        print("请安装PyTorch: pip install torch torchvision torchaudio")

def main():
    """主函数"""
    print("="*50)
    print("GPU问题检测和修复工具")
    print("="*50)
    
    check_system()
    check_pytorch()
    
    print("\n="*50)
    print("检测完成，建议执行以下步骤：")
    print("1. 如果NVIDIA驱动未安装，请先安装驱动")
    print("2. 如果CUDA工具包未安装，请安装CUDA")
    print("3. 如果PyTorch无法使用CUDA，请运行:")
    print("   python scripts/install_gpu_pytorch.py")
    print("4. 安装完成后，运行检测脚本验证:")
    print("   python scripts/check_gpu.py")
    print("="*50)

if __name__ == "__main__":
    main() 