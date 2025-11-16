import torch
import torch.nn as nn 

class SinusoidalEncoding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t):
        out = torch.zeros((len(t), self.emb_dim)) # (batch_size, emb_dim)
        idx = torch.arange(self.emb_dim).unsqueeze(0) # (1, emb_dim)

        t = t.unsqueeze(1) # (batch_size, 1)

        values =  t * (1 / (10000 ** (2 * (idx // 2)))) # (batch_size, emb_dim)

        out[:, 0::2] = torch.sin(values[:, 0::2])
        out[:, 1::2] = torch.cos(values[:, 1::2])

        return out # (batch_size, 512)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.gn1 = nn.GroupNorm(num_groups = 1, num_channels = out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.gn2 = nn.GroupNorm(num_groups = 1, num_channels = out_channels)
        

    def forward(self, x):
        return self.gn2(self.conv2(self.gelu(self.gn1(self.conv1(x)))))

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.out_channels = out_channels

        self.mp = nn.MaxPool2d(kernel_size = 2, padding = 0)
        self.conv_block1 = ConvBlock(in_channels, out_channels)
        self.conv_block2 = ConvBlock(out_channels, out_channels)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(emb_dim, out_channels)

    def forward(self, x, embeddings):

        # (batch_size, out_channels, downsampled_res, downsampled_res)
        out1 = self.conv_block2(self.conv_block1(self.mp(x))) 

        out2 = self.linear(self.silu(embeddings)) # (batch_size, out_channels)

        # (batch_size, out_channels, 1, 1)
        out2 = out2.reshape((*out2.shape[0:2], 1, 1)) 

        return out1 + out2 # (batch_size, out_channels, downsampled_res, downsampled_res)
 
class MHA(nn.Module):
    def __init__(self, num_channels: int, num_heads: int):
        super().__init__()
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.d_k = torch.tensor(num_channels // num_heads)

        self.ln = nn.LayerNorm(num_channels)
        self.W_q = nn.Linear(num_channels, num_channels)
        self.W_k = nn.Linear(num_channels, num_channels)
        self.W_v = nn.Linear(num_channels, num_channels)
        self.W_o = nn.Linear(num_channels, num_channels)

        self.fc = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels)
        )

    def forward(self, x):
        # (batch_size, img_size ** 2, num_channels)
        x = x.reshape(*x.shape[0:2], x.shape[2] * x.shape[3]).transpose(1, 2)

        norm_x = self.ln(x)

        # (batch_size, img_size ** 2, num_channels)
        q, k, v = self.W_q(norm_x), self.W_k(norm_x), self.W_v(norm_x)

        # (batch_size, num_heads, img_size ** 2, d_k)
        q = q.view(*q.shape[0:2], self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(*k.shape[0:2], self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(*v.shape[0:2], self.num_heads, self.d_k).transpose(1, 2)

        # (batch_size, num_heads, img_size ** 2, img_size ** 2)
        scores = (q @ k.transpose(-2, -1)) / torch.sqrt(self.d_k)

        # (batch_size, num_heads, img_size ** 2, d_k)
        mha_out = torch.softmax(scores, dim = -1) @ v

        # (batch_size, img_size ** 2, num_channels)
        mha_out = mha_out.transpose(1, 2).contiguous().view(mha_out.shape[0], -1, 
                                                            self.num_channels)

        mha_out = self.W_o(mha_out) # (batch_size, img_size ** 2, num_channels)

        fc_in = mha_out + x # (batch_size, img_size ** 2, num_channels)
        fc_out = self.fc(fc_in) # (batch_size, img_size ** 2, num_channels)

        out = mha_out + fc_out # (batch_size, img_size ** 2, num_channels)

        img_size = int(torch.sqrt(torch.tensor(out.shape[1])))

        # (batch_size, num_channels, img_size, img_size)
        return out.view(out.shape[0], -1, img_size, img_size)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int, out_channels: int,
                 emb_dims: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv1 = ConvBlock(in_channels, hid_channels)
        self.conv2 = ConvBlock(hid_channels, out_channels)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(emb_dims, out_channels)

    def forward(self, x_to_upsample, x, embeddings):

        # (batch_size, num_channels, img_size, img_size)
        upsampled_x = self.upsample(x_to_upsample) 

        # (batch_size, 2 * num_channels, img_size, img_size)
        x = torch.concat([x, upsampled_x], dim = 1) 

        out1 = self.conv2(self.conv1(x)) # (batch_size, out_channels, img_size, img_size)
        out2 = self.linear(self.silu(embeddings)) # (batch_size, out_channels)

        out2 = out2.view(*out2.shape, 1, 1) # (batch_size, out_channels, 1, 1)

        return out1 + out2 # (batch_size, out_channels, img_size, img_size)

class UNet(nn.Module):
    '''Strictly for (N, 3, 64, 64) image data'''
    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        self.encoding = SinusoidalEncoding(emb_dim)

        self.conv_block1 = ConvBlock(3, 64)
        self.conv_block2 = ConvBlock(512, 512)
        self.conv_block3 = ConvBlock(512, 512)
        self.conv_block4 = ConvBlock(512, 256)

        self.ds_block1 = DownSampleBlock(64, 128, emb_dim)
        self.ds_block2 = DownSampleBlock(128, 256, emb_dim)
        self.ds_block3 = DownSampleBlock(256, 512, emb_dim)

        self.us_block1 = UpsampleBlock(512, 256, 128, 512)
        self.us_block2 = UpsampleBlock(256, 128, 64, 512)
        self.us_block3 = UpsampleBlock(128, 64, 64, 512)

        self.mha1 = MHA(128, num_heads)
        self.mha2 = MHA(256, num_heads)
        self.mha3 = MHA(512, num_heads)
        self.mha4 = MHA(128, num_heads)
        self.mha5 = MHA(64, num_heads)
        self.mha6 = MHA(64, num_heads)

        self.final_conv = nn.Conv2d(64, 3, kernel_size = 3, padding = 1)

    def forward(self, x, timesteps):
        embeddings = self.encoding(timesteps)

        conv1_out = self.conv_block1(x)
        ds1_out = self.ds_block1(conv1_out, embeddings)
        mha1_out = self.mha1(ds1_out)
        ds2_out = self.ds_block2(mha1_out, embeddings)
        mha2_out = self.mha2(ds2_out)
        ds3_out = self.ds_block3(mha2_out, embeddings)
        mha3_out = self.mha3(ds3_out)

        us1_in = self.conv_block4(self.conv_block3(self.conv_block2(mha3_out)))
        skip_conn1 = mha2_out
        us1_out = self.us_block1(us1_in, skip_conn1, embeddings)

        us2_in = self.mha4(us1_out)
        skip_conn2 = mha1_out
        us2_out = self.us_block2(us2_in, skip_conn2, embeddings)

        us3_in = self.mha5(us2_out)
        skip_conn3 = conv1_out
        us3_out = self.us_block3(us3_in, skip_conn3, embeddings)

        return self.final_conv(us3_out)




