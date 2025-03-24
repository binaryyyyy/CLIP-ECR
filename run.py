"""
CLIP-ECR 一键运行脚本
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP-ECR 一键运行')
    
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'demo'], default='train',
                        help='运行模式：训练(train)、评估(evaluate)或演示(demo)')
    
    parser.add_argument('--image_dir', type=str, default='F:/1_ML/data/image',
                        help='NRRD图像文件目录')
    parser.add_argument('--label_file', type=str, default='F:/1_ML/data/table_info.xlsx',
                        help='标签Excel文件路径')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型检查点路径（评估和演示模式需要）')
    
    parser.add_argument('--image_path', type=str, default=None,
                        help='单张图像路径（仅演示模式需要）')
    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批量大小')
    
    parser.add_argument('--gpu', action='store_true',
                        help='强制使用GPU，如无可用GPU则报错')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()
    
    # 确保脚本目录存在
    scripts_dir = project_root / 'scripts'
    if not scripts_dir.exists():
        print(f"错误：找不到脚本目录 {scripts_dir}")
        return
    
    # 设置Python解释器路径
    python_executable = sys.executable
    
    # 根据模式选择要运行的脚本
    if args.mode == 'train':
        script_path = scripts_dir / 'train.py'
        
        cmd = [
            python_executable,
            str(script_path),
            f"--image_dir={args.image_dir}",
            f"--label_file={args.label_file}",
            f"--batch_size={args.batch_size}"
        ]
        
        # 如果指定了GPU，添加GPU参数
        if args.gpu:
            cmd.append("--gpu")
        
    elif args.mode == 'evaluate':
        script_path = scripts_dir / 'evaluate.py'
        
        if args.model_path is None:
            print("错误：评估模式需要指定 --model_path")
            return
        
        cmd = [
            python_executable,
            str(script_path),
            f"--image_dir={args.image_dir}",
            f"--label_file={args.label_file}",
            f"--model_path={args.model_path}",
            f"--batch_size={args.batch_size}"
        ]
        
        # 如果指定了GPU，添加GPU参数
        if args.gpu:
            cmd.append("--gpu")
        
    elif args.mode == 'demo':
        script_path = scripts_dir / 'demo.py'
        
        if args.model_path is None or args.image_path is None:
            print("错误：演示模式需要同时指定 --model_path 和 --image_path")
            return
        
        cmd = [
            python_executable,
            str(script_path),
            f"--image_path={args.image_path}",
            f"--model_path={args.model_path}"
        ]
        
        # 如果指定了GPU，添加GPU参数
        if args.gpu:
            cmd.append("--gpu")
    
    # 运行命令
    print(f"运行命令: {' '.join(cmd)}")
    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"运行时发生错误: {e}")


if __name__ == '__main__':
    main() 