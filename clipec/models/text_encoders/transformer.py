import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """
    基于Transformer的文本编码器
    使用预训练的BERT模型提取文本特征，并映射到与图像特征相同的嵌入空间
    """
    
    def __init__(self, model_name="bert-base-chinese", embedding_dim=1024, max_length=64):
        """
        初始化文本编码器
        
        Args:
            model_name (str): 预训练模型名称
            embedding_dim (int): 最终嵌入维度
            max_length (int): 输入文本的最大长度
        """
        super(TextEncoder, self).__init__()
        
        # 加载预训练的BERT模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # 提取的特征维度（BERT-base隐藏状态大小为768）
        self.feature_dim = self.transformer.config.hidden_size
        
        # 映射层（将BERT特征映射到指定的嵌入维度）
        self.projection = nn.Linear(self.feature_dim, embedding_dim)
        
        self.max_length = max_length
    
    def encode_text(self, text_list):
        """
        编码文本列表
        
        Args:
            text_list (list): 文本字符串列表
            
        Returns:
            torch.Tensor: 文本嵌入
        """
        # 分词
        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移动到与模型相同的设备
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 通过BERT提取特征
        with torch.no_grad():
            outputs = self.transformer(**inputs)
        
        # 使用[CLS]标记的输出作为文本表示
        text_features = outputs.last_hidden_state[:, 0, :]
        
        # 通过映射层
        text_embeddings = self.projection(text_features)
        
        return text_embeddings
    
    def forward(self, texts):
        """
        前向传播
        
        Args:
            texts (list): 文本字符串列表
            
        Returns:
            torch.Tensor: 文本嵌入
        """
        return self.encode_text(texts)


class SimpleTextEncoder(nn.Module):
    """更简单的文本编码器，使用预定义的词嵌入"""
    
    def __init__(self, vocab=None, embedding_dim=1024):
        """
        初始化简单文本编码器
        
        Args:
            vocab (dict): 词汇表，从词到索引的映射
            embedding_dim (int): 嵌入维度
        """
        super(SimpleTextEncoder, self).__init__()
        
        # 如果没有提供词汇表，则根据提供的表格数据创建一个默认词汇表
        if vocab is None:
            # 从表格数据中提取所有唯一标签值
            # 注意：假设数据加载时已进行清理（例如，去除多余空格，标准化特殊字符）
            self.vocab = {
                # Location (来自 'Primary Site - labeled')
                "Upper": 0,
                "Middle": 1,
                "Lower": 2,
                "Cervical": 3,
                "Abdominal": 4,

                # AJCC Stage Groups (来自 'Derived AJCC Stage Group, 7th ed' 和 'AJCC8th', 标准化后)
                "1a": 5,
                "1b": 6,
                "2a": 7,
                "2b": 8,
                "3a": 9,
                "3b": 10,
                "3c": 11,
                "4": 12,  # 涵盖 '4 ' 和 '4'
                "4a": 13,
                "4b": 14,

                # T/N/M Stages 和 Survival (来自 'Derived AJCC T', 'N', 'M' 和 'SEER cause-specific death')
                # 这些列主要使用数字字符串
                "0": 15,  # 可代表 T0(虽然未见), N0, M0, Survival=alive
                "1": 16,  # 可代表 T1(虽然表格中有1a/1b), N1, M1, Survival=dead
                "2": 17,  # 可代表 T2, N2
                "3": 18,  # 可代表 T3, N3
                # T4 已包含在 "4": 12

                # Padding Token
                "<PAD>": 19
            }
            # 验证：总共 5(loc) + 10(stage) + 4(num) + 1(pad) = 20 个词汇
        else:
            self.vocab = vocab
        
        # 创建嵌入层
        self.embedding = nn.Embedding(len(self.vocab), embedding_dim)
        
        # 简单的转换层
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        
    def tokenize(self, text):
        """
        将文本转换为索引
        """
        if text in self.vocab:
            return self.vocab[text]
        else:
            return self.vocab["<PAD>"]
    
    def forward(self, texts):
        """
        前向传播
        
        Args:
            texts (list): 文本字符串列表
            
        Returns:
            torch.Tensor: 文本嵌入
        """
        device = next(self.parameters()).device
        
        # 将文本转换为索引
        indices = [self.tokenize(text) for text in texts]
        indices = torch.tensor(indices, device=device)
        
        # 获取嵌入
        embeddings = self.embedding(indices)
        
        # 通过全连接层
        outputs = self.fc(embeddings)
        
        # 归一化
        outputs = F.normalize(outputs, p=2, dim=1)
        
        return outputs 