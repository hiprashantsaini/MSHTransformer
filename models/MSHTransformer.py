# MultiScaleHazeTransformer: MSHTransformer 
import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.size()

        x = F.adaptive_avg_pool2d(x, (h // 2, w // 2))  # Reduce resolution
        x = x.view(b, c, -1).permute(0, 2, 1)  # (B, H*W, C)
        
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff_out = self.feed_forward(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        x = x.permute(0, 2, 1).view(b, c, h // 2, w // 2)  # (B, C, H/2, W/2)
        x = F.interpolate(x, scale_factor=2)  # Restore original resolution
        return x

class MultiscaleChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiscaleChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
        self.conv5 = nn.Conv2d(in_channels, in_channels, 7, padding=3)

        # Channel attention components
        self.fc1 = nn.Linear(in_channels, in_channels // 8)
        self.fc2 = nn.Linear(in_channels // 8, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Spatial attention for each scale
        attn1 = F.relu(self.conv1(x))
        attn3 = F.relu(self.conv3(x))
        attn5 = F.relu(self.conv5(x))

        # Channel attention
        pooled1 = F.adaptive_avg_pool2d(attn1, (1, 1)).view(b, c)
        pooled3 = F.adaptive_avg_pool2d(attn3, (1, 1)).view(b, c)
        pooled5 = F.adaptive_avg_pool2d(attn5, (1, 1)).view(b, c)

        channel_weights1 = self.sigmoid(self.fc2(F.relu(self.fc1(pooled1)))).view(b, c, 1, 1)
        channel_weights3 = self.sigmoid(self.fc2(F.relu(self.fc1(pooled3)))).view(b, c, 1, 1)
        channel_weights5 = self.sigmoid(self.fc2(F.relu(self.fc1(pooled5)))).view(b, c, 1, 1)

        # Combine spatial and channel attention
        attn1 = attn1 * channel_weights1
        attn3 = attn3 * channel_weights3
        attn5 = attn5 * channel_weights5

        return attn1 + attn3 + attn5

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.ms_attention = MultiscaleChannelAttention(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x  # Residual connection
        res = self.conv2(res)
        res = self.ms_attention(res)  # Apply multiscale channel attention
        res += x  # Another residual connection
        return res

class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))  # Final convolution in the group
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x  # Residual connection
        return res

class MSHTransformer(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(MSHTransformer, self).__init__()
        self.gps = gps
        self.dim = 32
        kernel_size = 3
        
        # Pre-Processing Layer
        pre_process = [conv(3, self.dim, kernel_size)]
        self.pre = nn.Sequential(*pre_process)

        # Two Groups
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        
        # Adding a Transformer block after the groups
        self.transformer = TransformerBlock(dim=self.dim)

        # Post-Processing Layers
        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)
        ]
        self.post = nn.Sequential(*post_process)

    def forward(self, x1):
        x = self.pre(x1)  # Pre-Processing
        res1 = self.g1(x)  # First group
        res2 = self.g2(res1)  # Second group

        # Apply the Transformer block
        res2 = self.transformer(res2)

        if self.gps == 2:  # Handle two groups
            out = res1 + res2
        else:
            raise ValueError("Unsupported number of groups (gps).")
    
        x = self.post(out)  # Post-Processing
        return x + x1  # Skip connection with the input

# Testing the Network
if __name__ == "__main__":
    net = MSHTransformer(gps=2, blocks=10)
    print(net)