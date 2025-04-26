import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from ..utils.helpers import save_model, get_device


class CLIPTrainer:
    """CLIP模型训练器"""
    
    def __init__(
        self,
        model,
        learning_rate=1e-4,
        weight_decay=1e-4,
        save_dir="checkpoints",
        force_gpu=False
    ):
        """
        初始化训练器
        
        Args:
            model (nn.Module): CLIP模型
            learning_rate (float): 学习率
            weight_decay (float): 权重衰减
            save_dir (str): 保存模型的目录
            force_gpu (bool): 是否强制使用GPU
        """
        self.model = model
        self.device = get_device(force_gpu=force_gpu)
        self.model.to(self.device)
        
        print(f"模型已移至设备: {self.device}")
        # 打印模型大小
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数总数: {total_params:,}")
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            # TODO: 需要根据数据集的大小调整T_max 一般来说调整为epoch的一半
            T_max=25,
            eta_min=1e-6 # 最小学习率 防止学习率过低
        )
        
        # 保存目录
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True) # exist_ok=True 如果存在 不执行创建 跳过
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, data_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # 获取图像和对应标签
            images = batch['image'].to(self.device)
            
            # 获取文本描述（这里简化，实际应用中根据具体任务生成文本描述）
            # 使用AJCC分期作为文本描述
            texts = []
            # 检查labels的类型和结构
            batch_labels = batch['labels']
            
            # TODO: 需要修改并添加更多的标签
            # 如果是嵌套的字典，需要按以下方式处理
            if isinstance(batch_labels, dict):
                # 处理单个批次的情况
                if 'AJCC8th' in batch_labels and batch_labels['AJCC8th'] is not None:
                    text = f"食管癌AJCC分期{batch_labels['AJCC8th']}"
                else:
                    text = "未知分期"
                texts.append(text)
            else:
                # 处理多个样本的情况 - 可能是字典列表、字符串或其他类型
                for label_item in batch_labels:
                    # 如果是字典类型
                    if isinstance(label_item, dict):
                        if 'AJCC8th' in label_item and label_item['AJCC8th'] is not None:
                            text = f"食管癌AJCC分期{label_item['AJCC8th']}"
                        else:
                            text = "未知分期"
                    else:
                        # 如果不是字典，则直接使用标签作为文本
                        text = "未知分期"
                    texts.append(text)
            
            # 确保有足够的文本标签
            if len(texts) != images.size(0):
                print(f"文本数量不匹配图像数量: {len(texts)} != {images.size(0)}")
                # 如果文本数量不匹配图像数量，复制或截断到相同长度
                if len(texts) < images.size(0):
                    texts = texts * (images.size(0) // len(texts) + 1)
                texts = texts[:images.size(0)]
            
            # 前向传播
            # nn.Module 基类实现了 __call__ 方法，使得我们可以直接调用模型对象（如 model(input)）
            # 因此，可以直接调用 self.model(images, texts) 来调用forward方法
            _, _, similarity = self.model(images, texts)
            
            # 计算损失
            loss = self.model.compute_loss(similarity)
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新进度条
            current_loss = loss.item()
            total_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"}) # 更新进度条 在尾部显示当前损失
        
        # 更新学习率
        self.scheduler.step()
        
        avg_loss = total_loss / len(data_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, data_loader):
        """在验证集上评估模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Validating"):
                # 获取图像和对应标签
                images = batch['image'].to(self.device)
                
                # 获取文本描述
                texts = []
                # 检查labels的类型和结构
                batch_labels = batch['labels']
                
                # 如果是嵌套的字典，需要按以下方式处理
                if isinstance(batch_labels, dict):
                    # 处理单个批次的情况
                    if 'AJCC8th' in batch_labels and batch_labels['AJCC8th'] is not None:
                        text = f"食管癌AJCC分期{batch_labels['AJCC8th']}"
                    else:
                        text = "未知分期"
                    texts.append(text)
                else:
                    # 处理多个样本的情况 - 可能是字典列表、字符串或其他类型
                    for label_item in batch_labels:
                        # 如果是字典类型
                        if isinstance(label_item, dict):
                            if 'AJCC8th' in label_item and label_item['AJCC8th'] is not None:
                                text = f"食管癌AJCC分期{label_item['AJCC8th']}"
                            else:
                                text = "未知分期"
                        else:
                            # 如果不是字典，则直接使用标签作为文本
                            text = "未知分期"
                        texts.append(text)
                
                # 确保有足够的文本标签
                if len(texts) != images.size(0):
                    # 如果文本数量不匹配图像数量，复制或截断到相同长度
                    if len(texts) < images.size(0):
                        texts = texts * (images.size(0) // len(texts) + 1)
                    texts = texts[:images.size(0)]
                
                # 前向传播
                _, _, similarity = self.model(images, texts)
                
                # 计算损失
                loss = self.model.compute_loss(similarity)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs=50, save_best=True):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs (int): 训练轮数
            save_best (bool): 是否只保存最佳模型
            
        Returns:
            dict: 训练历史记录
        """
        print(f"开始训练 - 使用设备: {self.device}")
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # 验证
            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch}/{num_epochs} - Validation Loss: {val_loss:.4f}")
            
            # 保存模型
            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_path = save_model(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    None,
                    self.save_dir,
                    "best_model.pt"
                )
                print(f"保存最佳模型到 {save_path}")
            
            # 每10个epoch保存一次
            if epoch % 10 == 0:
                save_path = save_model(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    None,
                    self.save_dir,
                    f"model_epoch_{epoch}.pt"
                )
                print(f"保存checkpoint到 {save_path}")
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 返回训练历史
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def plot_training_curves(self):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        curves_path = os.path.join(self.save_dir, "training_curves.png")
        plt.savefig(curves_path)
        plt.close()
        print(f"训练曲线已保存到 {curves_path}")
    
    def predict(self, data_loader, text_labels):
        """
        使用训练好的模型进行预测
        
        Args:
            data_loader: 测试数据加载器
            text_labels (list): 文本标签列表
            
        Returns:
            tuple: (预测结果, 真实标签)
        """
        self.model.eval() # 设置为评估模式
        
        all_image_embeddings = []
        all_labels = []
        all_patient_ids = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                images = batch['image'].to(self.device)
                
                # 获取图像嵌入
                image_embeddings = self.model.encode_image(images)
                all_image_embeddings.append(image_embeddings)
                
                # 保存标签和患者ID
                batch_labels = batch['labels']
                
                # 处理不同格式的标签
                if isinstance(batch_labels, dict):
                    # 单个样本的情况
                    all_labels.append(batch_labels)
                else:
                    # 多个样本的情况
                    for label_item in batch_labels:
                        if isinstance(label_item, dict):
                            all_labels.append(label_item)
                        else:
                            # 如果不是字典类型，创建一个包含此值的字典
                            all_labels.append({"value": label_item})
                
                # 处理患者ID
                if isinstance(batch['patient_id'], list):
                    # 将列表中的元素分开添加到all_patient_ids列表中 而不是在末尾添加一个列表对象
                    all_patient_ids.extend(batch['patient_id'])
                else:
                    all_patient_ids.append(batch['patient_id'])
        
        # 合并所有嵌入
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        
        # 为所有可能的标签创建文本嵌入
        text_embeddings = self.model.encode_text(text_labels)
        
        # 计算相似度
        similarity = torch.matmul(all_image_embeddings, text_embeddings.t())
        
        # 获取最大相似度的索引
        max_indices = torch.argmax(similarity, dim=1).cpu().numpy()
        
        # 转换为预测标签
        predictions = [text_labels[i] for i in max_indices]
        
        return predictions, all_labels, all_patient_ids 