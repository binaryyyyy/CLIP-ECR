"""
GPU可用性检查脚本
用于诊断PyTorch是否能正确识别GPU
"""

import torch
import sys

def check_gpu():
    print("Python版本:", sys.version)
    print("PyTorch版本:", torch.__version__)
    print("CUDA是否可用:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA版本:", torch.version.cuda)
        device_count = torch.cuda.device_count()
        print("GPU设备数量:", device_count)
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            print(f"GPU {i}: {device_name} (CUDA能力: {device_capability[0]}.{device_capability[1]})")
            
            # 显示当前GPU内存使用情况
            print(f"GPU {i} 内存总量: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")
            print(f"GPU {i} 当前已分配内存: {torch.cuda.memory_allocated(i) / 1024 / 1024 / 1024:.2f} GB")
            print(f"GPU {i} 当前缓存内存: {torch.cuda.memory_reserved(i) / 1024 / 1024 / 1024:.2f} GB")
            
        # 尝试创建一个张量在GPU上
        try:
            x = torch.tensor([1, 2, 3], device="cuda")
            print("成功在GPU上创建张量:", x)
        except Exception as e:
            print("在GPU上创建张量失败:", e)
    else:
        print("PyTorch未检测到CUDA设备。可能的原因:")
        print("1. 未安装NVIDIA GPU驱动")
        print("2. 未安装CUDA工具包")
        print("3. 安装了CPU版本的PyTorch")
        print("4. CUDA版本与PyTorch不兼容")
        
        # 如果有NVIDIA GPU但PyTorch不能检测到，尝试获取系统信息
        try:
            import subprocess
            import platform
            
            print("\n系统信息:")
            print("操作系统:", platform.platform())
            
            if platform.system() == "Windows":
                # 在Windows上尝试获取GPU信息
                try:
                    nvidia_smi_output = subprocess.check_output("nvidia-smi", shell=True, stderr=subprocess.STDOUT).decode()
                    print("\nNVIDIA-SMI输出:")
                    print(nvidia_smi_output)
                except subprocess.CalledProcessError:
                    print("无法运行nvidia-smi，可能未安装NVIDIA驱动或驱动不正常")
        except Exception as e:
            print("获取系统信息失败:", e)

if __name__ == "__main__":
    check_gpu() 