import os
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from ..utils.helpers import save_model, get_device


class CLIPTrainer:
    """CLIP模型训练器"""
    
    def __init__(
        self,
        model,
        learning_rate=5e-4,
        weight_decay=5e-5,
        save_dir="checkpoints",
        force_gpu=True,
        use_mixed_precision=False  # 添加混合精度训练选项
    ):
        """
        初始化训练器
        
        Args:
            model (nn.Module): CLIP模型
            learning_rate (float): 学习率
            weight_decay (float): 权重衰减
            save_dir (str): 保存模型的目录
            force_gpu (bool): 是否强制使用GPU
            use_mixed_precision (bool): 是否使用混合精度训练
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
        
        # 学习率调度器 - 使用更激进的学习率衰减
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,  # 每次减半
            patience=2   # 2个epoch没改善就降低学习率
        )
        
        # 打印初始学习率
        print(f"初始学习率: {self.optimizer.param_groups[0]['lr']}")
        
        # 混合精度训练设置
        self.use_mixed_precision = use_mixed_precision
        if use_mixed_precision and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("已启用混合精度训练")
        
        # 保存目录
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)  # exist_ok=True 如果存在不执行创建跳过
        
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
            # print(f"batch_labels: {batch_labels}") # 调试用
            
            # TODO: 需要根据实际需求添加更多标签的处理逻辑
            # 从 batch_labels 字典中获取 'AJCC8th' 标签列表
            # batch_labels 的结构是 {'label_name': [label1, label2, ...]}
            ajcc_labels = batch_labels.get('AJCC8th', [])
            num_samples = images.size(0)  # 获取批次中的样本数量

            # 确保 ajcc_labels 列表长度与批次大小一致
            # 如果标签列表长度不足，用默认值（例如空字符串）填充
            if len(ajcc_labels) < num_samples:
                print(
                    f"警告: AJCC8th 标签数量 ({len(ajcc_labels)}) 少于批次大小 "
                    f"({num_samples})。将用'未知分期'填充。"
                )
                ajcc_labels.extend([""] * (num_samples - len(ajcc_labels)))
            elif len(ajcc_labels) > num_samples:
                # 如果标签列表过长（理论上不应发生），截断
                print(
                    f"警告: AJCC8th 标签数量 ({len(ajcc_labels)}) 多于批次大小 "
                    f"({num_samples})。将截断标签列表。"
                )
                ajcc_labels = ajcc_labels[:num_samples]

            # 为批次中的每个样本生成文本描述
            for i in range(num_samples):
                label = ajcc_labels[i]
                # 检查标签是否有效（非空字符串且非None/NaN）
                # 使用 str(label) 来处理可能的数字或其他类型，并检查是否为空
                label_str = (
                    str(label).strip()
                    if label is not None and pd.notna(label)
                    else ""
                )

                if label_str:
                    text = f"食管癌AJCC分期{label_str}"
                else:
                    text = "未知分期"
                texts.append(text)
            
            # 确保有足够的文本标签
            if len(texts) != images.size(0):
                print(f"文本数量不匹配图像数量: {len(texts)} != {images.size(0)}")
                # 如果文本数量不匹配图像数量，复制或截断到相同长度
                if len(texts) < images.size(0):
                    texts = texts * (images.size(0) // len(texts) + 1)
                texts = texts[:images.size(0)]
            
            # 前向传播 - 使用混合精度训练
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    _, _, similarity = self.model(images, texts)
                    loss = self.model.compute_loss(similarity)
                
                # 使用scaler进行反向传播和优化
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准训练过程
                _, _, similarity = self.model(images, texts)
                loss = self.model.compute_loss(similarity)
                loss.backward()
                self.optimizer.step()
            
            # 更新进度条
            current_loss = loss.item()
            total_loss += current_loss
            # 更新进度条 在尾部显示当前损失
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
        
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
                    if ('AJCC8th' in batch_labels
                            and batch_labels['AJCC8th'] is not None):
                        text = f"食管癌AJCC分期{batch_labels['AJCC8th']}"
                    else:
                        text = "未知分期"
                    texts.append(text)
                else:
                    # 处理多个样本的情况 - 可能是字典列表、字符串或其他类型
                    for label_item in batch_labels:
                        # 如果是字典类型
                        if isinstance(label_item, dict):
                            if ('AJCC8th' in label_item
                                    and label_item['AJCC8th'] is not None):
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
                
                # 前向传播（验证时不需要使用混合精度）
                _, _, similarity = self.model(images, texts)
                
                # 计算损失
                loss = self.model.compute_loss(similarity)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs=10, save_best=True):
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
        
        # 添加早停计数器和阈值
        patience = 5
        patience_counter = 0
        
        # 记录学习率
        lr_history = []
        
        for epoch in range(1, num_epochs + 1):
            # 当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            print(f"当前学习率: {current_lr:.6f}")
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # 验证
            val_loss = self.validate(val_loader)
            print(
                f"Epoch {epoch}/{num_epochs} - Validation Loss: {val_loss:.4f}"
            )
            
            # 更新学习率调度器
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # 如果学习率改变，打印日志
            if old_lr != new_lr:
                print(f"学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")
            
            # 保存模型
            if save_best and val_loss < self.best_val_loss:
                # 如果验证损失改善，重置早停计数器
                patience_counter = 0
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
            elif save_best:
                # 验证损失没有改善，增加早停计数器
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"验证损失连续{patience}个epoch未改善，提前停止训练")
                    break
            
            # 每5个epoch保存一次，更频繁地保存checkpoint
            if epoch % 5 == 0:
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
        self.plot_training_curves(lr_history)
        
        # 返回训练历史
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'lr_history': lr_history
        }
    
    def plot_training_curves(self, lr_history):
        """绘制训练和验证损失曲线"""
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制损失曲线
        ax1.plot(self.train_losses, label='Training Loss', marker='o')
        ax1.plot(self.val_losses, label='Validation Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制学习率曲线
        ax2.plot(lr_history, label='Learning Rate', marker='x', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')  # 使用对数刻度更好地显示学习率变化
        ax2.grid(True)
        
        plt.tight_layout()
        
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
        self.model.eval()  # 设置为评估模式
        
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