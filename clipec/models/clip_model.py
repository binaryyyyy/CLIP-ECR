import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoders.resnet import resnet50, resnet152
from .text_encoders.transformer import TextEncoder, SimpleTextEncoder


class CLIPModel(nn.Module):
    """
    CLIP模型实现
    结合图像编码器和文本编码器，计算它们之间的相似度
    """
    
    def __init__(
        self, 
        image_encoder_name="resnet50", 
        text_encoder_name="simple",
        embedding_dim=1024,
        temperature=0.1
    ):
        """
        初始化CLIP模型
        
        Args:
            image_encoder_name (str): 图像编码器名称
            text_encoder_name (str): 文本编码器名称
            embedding_dim (int): 嵌入维度
            temperature (float): 相似度计算的温度系数
        """
        super(CLIPModel, self).__init__()
        
        # 初始化图像编码器
        if image_encoder_name == "resnet50":
            # 加载预训练的ResNet，移除最后的分类层
            # TODO: 更改使用weights参数 指定特定的权重
            self.image_encoder = resnet152(pretrained=True)
            # 修改最后的全连接层以输出所需维度的特征
            in_features = self.image_encoder.fc.in_features # 获取全连接层的输入维度
            self.image_encoder.fc = nn.Linear(in_features, embedding_dim) # 修改全连接层 embedding_dim：输出维度（向量化维度）
        else:
            raise ValueError(f"不支持的图像编码器: {image_encoder_name}")
        
        # 初始化文本编码器
        if text_encoder_name == "transformer":
            self.text_encoder = TextEncoder(embedding_dim=embedding_dim)
        elif text_encoder_name == "simple":
            self.text_encoder = SimpleTextEncoder(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"不支持的文本编码器: {text_encoder_name}")
        
        # 温度参数（控制softmax的平滑度）
        # temperature 用 nn.Parameter 包装成可训练的参数
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        
        # 初始化投影层
        self.image_projection = nn.Linear(embedding_dim, embedding_dim)
        self.text_projection = nn.Linear(embedding_dim, embedding_dim)
    
    def encode_image(self, images):
        """
        编码图像
        
        Args:
            images (torch.Tensor): 图像张量，形状为 [batch_size, 3, H, W]
            
        Returns:
            torch.Tensor: 图像特征，形状为 [batch_size, embedding_dim]
        """
        features = self.image_encoder(images)
        features = self.image_projection(features)
        return F.normalize(features, p=2, dim=1)
    
    def encode_text(self, texts):
        """
        编码文本
        
        Args:
            texts (list): 文本字符串列表
            
        Returns:
            torch.Tensor: 文本特征，形状为 [batch_size, embedding_dim]
        """
        features = self.text_encoder(texts)
        features = self.text_projection(features)
        return F.normalize(features, p=2, dim=1)
    
    def forward(self, images, texts):
        """
        前向传播，计算图像和文本之间的相似度
        
        Args:
            images (torch.Tensor): 图像张量
            texts (list): 文本字符串列表
            
        Returns:
            tuple: (图像特征, 文本特征, 相似度矩阵)
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        
        # 计算相似度矩阵 [batch_size, batch_size]
        # 每个图像与每个文本之间的余弦相似度
        similarity = torch.matmul(image_features, text_features.t()) / self.temperature
        
        return image_features, text_features, similarity
    
    def compute_loss(self, similarity, labels=None):
        """
        计算对比损失
        
        Args:
            similarity (torch.Tensor): 相似度矩阵
            labels (torch.Tensor, optional): 标签。如果为None，则假设每个图像与对应索引的文本匹配
            
        Returns:
            torch.Tensor: 计算得到的损失值
        """
        batch_size = similarity.size(0)
        
        # 如果没有提供标签，默认使用对角线作为正样本
        if labels is None:
            labels = torch.arange(batch_size, device=similarity.device)
        
        # 图像到文本的损失（每行应该最大值在对应的索引）
        image_loss = F.cross_entropy(similarity, labels)
        
        # 文本到图像的损失（每列应该最大值在对应的索引）
        text_loss = F.cross_entropy(similarity.t(), labels)
        
        # 加权总损失，增加图像到文本的权重
        # 修改损失权重，强调图像→文本的匹配
        total_loss = (0.7 * image_loss + 0.3 * text_loss)
        
        return total_loss 