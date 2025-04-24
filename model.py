import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)

from transformers import AutoModelForImageClassification

import torch
from torch import nn

class ICDModel(nn.Module):
    def __init__(self, unet_checkpoint_path, vae_checkpoint_path, vit_checkpoint_path):
        super().__init__()
        
        
        self.scale_factor = 0.18215

        self.unet = UNet2DConditionModel.from_pretrained(
            unet_checkpoint_path, use_safetensors=False
        )
        
        self.vae = AutoencoderKL.from_pretrained(
            vae_checkpoint_path, use_safetensors=False
        )
        # 锁定VAE的参数
        for param in self.vae.encoder.parameters():
            param.requires_grad = False
        for param in self.vae.quant_conv.parameters():
            param.requires_grad = False
            
        self.contidion = ConditionEmbedding(vit_checkpoint_path)
    
        self.classifier = nn.Sequential(
            nn.Linear(4 * 28 * 28, 128),
            nn.SiLU(),
            nn.Linear(128, 5),
        )

    def encoder_image(self, rgb_in: torch.Tensor) -> torch.Tensor:
        self.vae.eval()
        self.vae.quant_conv.eval()
        # 不需要梯度
        with torch.no_grad():
            # encode
            h = self.vae.encoder(rgb_in)
            moments = self.vae.quant_conv(h)
            mean, logvar = torch.chunk(moments, 2, dim=1)
            # scale latent
            rgb_latent = mean * self.scale_factor
        return rgb_latent
    
    def forward(self, rgb_in: torch.Tensor, ori_images) -> torch.Tensor:
        # 1. 编码
        rgb_latent = self.encoder_image(rgb_in)

        context = self.contidion(ori_images)

        time_step = torch.zeros(rgb_latent.shape[0], device=rgb_latent.device)

        latent = self.unet(sample=rgb_latent, timestep=time_step, encoder_hidden_states=context).sample

        # latent 的 shape 是 [batch_size, 4, 28, 28]
        # 展平为 [batch_size, 4*28*28]
        latent = latent.reshape(latent.shape[0], -1)

        return self.classifier(latent)
   

class ConditionEmbedding(nn.Module):
    def __init__(self, vit_checkpoint_path, embedding_dim=768):
        super().__init__()
        self.vit = AutoModelForImageClassification.from_pretrained(vit_checkpoint_path)
        # 冻结vit的参数
        for param in self.vit.parameters():
            param.requires_grad = False

        self.linear = nn.Sequential(
            nn.Linear(1000, 100),
            nn.SiLU(),
        )
        self.embeddings = nn.Embedding(100, embedding_dim)
        self.gamga = nn.Parameter(torch.ones(1, embedding_dim))
        self.output = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        
        x = self.vit(x).logits
        # 扩充一个维度
        x = x.unsqueeze(1)
        x = self.linear(x)
        x = x @ self.embeddings.weight
        x = x + self.gamga * self.output(x)
        return x
    
if __name__ == "__main__":
    model = ICDModel(
        unet_checkpoint_path="checkpoints/tiny_sd/unet",
        vae_checkpoint_path="checkpoints/tiny_sd/vae",
        vit_checkpoint_path="google/vit-base-patch16-224",
    )

    rgb = torch.randn(2, 3, 224, 224)
    ori_images = torch.randn(2, 3, 224, 224)

    output = model(rgb, ori_images)
    print(output.shape)  # 应该是 [2, 5]