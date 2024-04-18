import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        self.pather = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.positional_embedding = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.pather(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x += self.positional_embedding
        x = self.dropout(x)
        return x


class ViT(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout,
                 num_heads, activation, num_encoders, num_classes):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                        dropout=dropout, activation=activation,
                                                        batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_encoders)
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.MLP(x[:, 0, :])
        return x
