"""
安装GPU支持版本的PyTorch
根据当前系统的CUDA版本自动安装兼容的PyTorch
"""

import subprocess
import sys
import platform
import re
import os

def get_cuda_version():
    """获取系统CUDA版本"""
    try:
        # 尝试通过nvidia-smi获取CUDA版本
        output = subprocess.check_output("nvidia-smi", shell=True).decode()
        match = re.search(r"CUDA Version: (\d+\.\d+)", output)
        if match:
            return match.group(1)
        else:
            return None
    except Exception:
        return None

def install_gpu_pytorch():
    """安装GPU支持版本的PyTorch"""
    print("开始安装GPU支持版本的PyTorch...")
    
    # 检查Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"当前Python版本: {python_version}")
    
    # 获取CUDA版本
    cuda_version = get_cuda_version()
    if cuda_version:
        cuda_major_version = cuda_version.split('.')[0]
        print(f"检测到CUDA版本: {cuda_version} (主版本: {cuda_major_version})")
    else:
        print("未检测到CUDA，请确保已安装NVIDIA驱动和CUDA工具包")
        return
    
    # 根据CUDA版本选择PyTorch安装命令
    if platform.system() == "Windows":
        # Windows系统
        if cuda_major_version == "12":
            cmd = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        elif cuda_major_version == "11":
            cmd = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        elif cuda_major_version == "10":
            cmd = f"pip install torch==1.13.1+cu101 torchvision==0.14.1+cu101 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu101"
        else:
            print(f"不支持的CUDA版本: {cuda_version}，请手动安装对应版本的PyTorch")
            return
    else:
        # Linux/Mac系统
        if cuda_major_version == "12":
            cmd = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        elif cuda_major_version == "11":
            cmd = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        else:
            print(f"不支持的CUDA版本: {cuda_version}，请手动安装对应版本的PyTorch")
            return
    
    # 执行安装
    print(f"执行安装命令: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("PyTorch安装完成!")
        print("请重新运行check_gpu.py脚本验证GPU是否可用")
    except subprocess.CalledProcessError as e:
        print(f"安装失败: {e}")

if __name__ == "__main__":
    # 检查是否已安装PyTorch
    try:
        import torch
        print(f"当前已安装PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print("PyTorch已经可以使用GPU，无需重新安装")
            sys.exit(0)
        else:
            print("PyTorch无法使用GPU，将尝试重新安装...")
    except ImportError:
        print("未安装PyTorch，将进行安装...")
    
    # 询问用户是否继续
    response = input("是否继续安装GPU版本的PyTorch? [y/N]: ")
    if response.lower() == 'y':
        install_gpu_pytorch()
    else:
        print("安装已取消") 