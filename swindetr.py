
#!pip install swin-transformer-pytorch
import torch
from swin_transformer_pytorch import SwinTransformer
from torch import nn

class SwinDetr(nn.Module):
    """
    Demo Swin DETR implementation.
    """
    def __init__(self, num_classes, hidden_dim=384, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.backbone = nn.Sequential(
            *list(SwinTransformer(
            hidden_dim=96,
            layers=(2, 2, 6, 2),
            heads=(3, 6, 12, 24),
            channels=3,
            num_classes=3,
            head_dim=32,
            window_size=7,
            downscaling_factors=(4, 2, 2, 2),
            relative_pos_embedding=True
            ).children())        
            )[:-2]

        # create conversion layer
        self.conv = nn.Conv2d(384, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
            )

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone(inputs)
 
        temp = f" Shapes \n Output of Backbone : {x.shape}"

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        temp += f" \n Output of Conv : {h.shape}"

        # construct positional encodings
        H, W = h.shape[-2:]

        temp += f" \n H,W : {H, W}"
        temp += f" \n Col Embed Alone : {self.col_embed[:W].shape}"
        temp += f" \n Col Embed After : {self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1).shape}"
        temp += f" \n Row Embed Alone : {self.row_embed[:H].shape}"
        temp += f" \n Row Embed After : {self.row_embed[:H].unsqueeze(1).repeat(1, W, 1).shape}"
        temp += f" \n Cat Alone : {torch.cat([ self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),], dim=-1).shape}"

        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        temp += f" \n Cat After : {pos.shape}"
        temp += f" \n h.flatten(2).permute(2, 0, 1) : {h.flatten(2).permute(2, 0, 1).shape}"
        temp += f" \n self.query_pos.unsqueeze(1) : {self.query_pos.unsqueeze(1).shape}"

        

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        temp += f" \n h last : {h.shape}"

        print(temp)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}