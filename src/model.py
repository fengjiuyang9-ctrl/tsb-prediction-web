import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class E2Model(nn.Module):
    """
    E2: pretrained EfficientNet-B0 + metadata two-layer MLP + late fusion regression.
    """

    def __init__(
        self,
        meta_input_dim: int,
        meta_hidden: int = 32,
        meta_out: int = 32,
        head_hidden: int = 128,
        dropout: float = 0.2,
        use_imagenet_pretrained: bool = True,
    ):
        super().__init__()
        # 训练默认使用 ImageNet 预训练；推理可关闭以避免离线下载。
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if use_imagenet_pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        cnn_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_input_dim, meta_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden, meta_out),
            nn.ReLU(inplace=True),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(cnn_dim + meta_out, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(image)
        meta_feat = self.meta_mlp(meta)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        pred = self.reg_head(fused).squeeze(1)
        return pred

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
