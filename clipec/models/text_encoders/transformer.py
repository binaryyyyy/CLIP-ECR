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
        
        # 如果没有提供词汇表，创建一个简单的默认词汇表
        if vocab is None:
            self.vocab = {
                "Upper": 0, "Middle": 1, "Lower": 2,  # 食管位置
                "1": 3, "2a": 4, "2b": 5, "3a": 6, "3b": 7, "4": 8,  # AJCC分期
                "T1": 9, "T2": 10, "T3": 11, "T4": 12,  # T分期
                "N0": 13, "N1": 14, "N2": 15, "N3": 16,  # N分期
                "M0": 17, "M1": 18,  # M分期
                "alive": 19, "dead": 20,  # 生存状态
                "<PAD>": 21
            }
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