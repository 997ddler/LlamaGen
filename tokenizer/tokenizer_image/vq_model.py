# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as tdist
from sklearn.cluster import KMeans
import math

@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True # Original llamagen is True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0
    
    soft_l2_loss: bool = False
    kmeans_ini_flag: bool = False
    
    soft_ae_training: bool = False
    soft_ae_iters: int = 25000
    soft_ae_scheduler: str = 'cosine'
    
    soft_loss: bool = True
    lambda_loss: float = 0.5
    soft_representation: bool = True
    
    
    simvq_training: bool = False
    

    

class AEModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        
        if config.codebook_l2_norm:
            print('codebook_l2_norm is True')
            if config.soft_l2_loss:
                print('soft_l2_loss is True')
            else:
                print('soft_l2_loss is False')
        
        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        
        
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        
    def soft_l2_loss(self, h):
        # h shape: [B, C, H, W]
        h_reshaped = h.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        norm = torch.norm(h_reshaped, p=2, dim=-1)  # [B, H, W]
        target_norm = 1.0
        reg_loss = torch.mean((norm - target_norm) ** 2)
        return reg_loss  
    
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        
        
        reg_loss = 0.0
        if self.config.codebook_l2_norm: 
            if  self.config.soft_l2_loss:
                reg_loss = self.soft_l2_loss(h)
            else:
                h = F.normalize(h, p=2, dim=-1)
                reg_loss = 0.0
        
        return h, reg_loss
    
    def decode(self, h):
        h = self.post_quant_conv(h)
        h = self.decoder(h)

        return h
    
    def forward(self, x):
        h, loss = self.encode(x)
        dec = self.decode(h)
        return dec, loss

    @torch.no_grad()
    def img_to_reconstructed_img(self, x):
        h, _ = self.encode(x)
        dec = self.decode(h)
        return dec




class VQModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)


        self.quantize = VectorQuantizer(n_e=config.codebook_size, e_dim=config.codebook_embed_dim, 
                                        beta=config.commit_loss_beta, entropy_loss_ratio=config.entropy_loss_ratio,
                                        l2_norm=config.codebook_l2_norm, show_usage=config.codebook_show_usage, kmeans_ini_flag=config.kmeans_ini_flag,
                                        soft_ae_training=config.soft_ae_training, soft_ae_iters=config.soft_ae_iters, soft_ae_scheduler=config.soft_ae_scheduler, 
                                        simvq_training=config.simvq_training, lambda_loss=config.lambda_loss, soft_representation=config.soft_representation, soft_loss=config.soft_loss)
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input) 
        dec = self.decode(quant)
        return dec, diff
    
    @torch.no_grad()
    def img_to_reconstructed_img(self, x):
        quant, _, _ = self.encode(x)
        dec = self.decode(quant)
        return dec
    
    def get_codebook(self):
        return self.quantize.get_codebook()



class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, 
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions-1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        
        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch*ch_mult[self.num_resolutions-1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

       # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight
    
    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    def __init__(
                self, 
                n_e, 
                e_dim, 
                beta, 
                entropy_loss_ratio, 
                l2_norm, show_usage, 
                kmeans_ini_flag, 
                soft_ae_training=False, 
                soft_ae_iters=10000, 
                soft_ae_scheduler='cosine', 
                simvq_training=False, 
                lambda_loss=0.5, 
                soft_loss=True, 
                soft_representation=True
                ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.simvq_training = simvq_training

        self.kmeans_ini_flag = kmeans_ini_flag

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))
        
        
        if self.simvq_training:
            print('simvq_training is True')
            nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)
            for p in self.embedding.parameters():
                p.requires_grad = False
            self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)
        
        
        self.register_buffer('initialized_buffer', torch.zeros(1, dtype=torch.bool))
        
        self.soft_ae_training = soft_ae_training
        self.soft_ae_iters = soft_ae_iters
        self.soft_ae_scheduler = soft_ae_scheduler
        
        if self.soft_ae_training:
            self.sche_a = 1.0
            self.lambda_loss = lambda_loss
            self.soft_loss = soft_loss
            self.soft_representation = soft_representation

    @torch.no_grad()
    def init_codebook_with_first_batch(self, data):
        if self.initialized_buffer.item():
            return

        if tdist.get_rank() == 0:
            data = data.reshape(-1, self.e_dim)
            
            if isinstance(data, torch.Tensor):
                data = data.detach().float().cpu().numpy()
            print('begin initialize')
            kmeans = KMeans(n_clusters=self.n_e, random_state=0).fit(data)
            centroids = torch.from_numpy(kmeans.cluster_centers_).to(self.embedding.weight.device)
            print('finish initialize')
            self.embedding.weight.data.copy_(centroids)
            self.initialized_buffer.fill_(True)

    def update_sche_a(self, cur_iter):
        if cur_iter >= self.soft_ae_iters:
            self.sche_a = 0.0
            return
        
        progress = cur_iter / self.soft_ae_iters
        
        if self.soft_ae_scheduler == 'cosine':
            self.sche_a = 0.5 * (1 + math.cos(math.pi * progress))
        elif self.soft_ae_scheduler == 'linear':
            self.sche_a = 1 - progress
        elif self.soft_ae_scheduler == 'exponential':
            self.sche_a = math.exp(-5 * progress)
        else:
            raise ValueError(f"Invalid scheduler: {self.soft_ae_scheduler}")

    
    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.training and not self.initialized_buffer.item() and self.kmeans_ini_flag:
            rank = tdist.get_rank()
            world_size = tdist.get_world_size()
            
            z_flattened_detached = z_flattened.detach()
            
            gathered_list = [torch.zeros_like(z_flattened_detached) for _ in range(world_size)]
            
            tdist.barrier()
            tdist.all_gather(gathered_list, z_flattened_detached)
            
            if rank == 0:
                print('Initializing codebook with first training batch')
                concatenated_f_BChw = torch.cat(gathered_list, dim=0)
                self.init_codebook_with_first_batch(concatenated_f_BChw)
            
            tdist.barrier() 

            # Broadcast the initialized weights and flag to all processes
            tdist.broadcast(self.embedding.weight.data, 0)
            tdist.broadcast(self.initialized_buffer, 0)
            
            tdist.barrier()


        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            
            if self.simvq_training:
                quant_codebook = self.embedding_proj(self.embedding.weight)
                embedding = F.normalize(quant_codebook, p=2, dim=-1)
            else:
                embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            if self.simvq_training:
                quant_codebook = self.embedding_proj(self.embedding.weight)
                embedding = quant_codebook
            else:
                embedding = self.embedding.weight

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)
        
        
        # compute perplexity
        min_encodings = torch.zeros(
            min_encoding_indices.unsqueeze(1).shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
        
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        #########################################################
        
        # perplexity = None
        # min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        if self.training and self.soft_ae_training:
            if self.soft_representation:
                z_q = self.sche_a * z + (1-self.sche_a) * (z + (z_q - z).detach())
            else:
                if self.sche_a != 0.0:
                    z_q = z
                else:
                    z_q = z + (z_q - z).detach()
            
            if self.soft_loss:
                commit_loss = ((1 - self.lambda_loss) * (1 - self.sche_a) + self.lambda_loss) * commit_loss
                vq_loss = ((1 - self.lambda_loss) * (1 - self.sche_a) + self.lambda_loss) * vq_loss
            
        else: 
            # preserve gradients
            z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            if self.simvq_training:
                embedding = F.normalize(self.embedding_proj(self.embedding.weight), p=2, dim=-1)
            else:
                embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            if self.simvq_training:
                embedding = self.embedding_proj(self.embedding.weight)
            else:
                embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q

    def get_codebook(self):
        if self.simvq_training:
            return self.embedding_proj(self.embedding.weight)
        else:
            return self.embedding.weight


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def AE(**kwargs):
    return AEModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

VQ_models = {'VQ-16': VQ_16, 'VQ-8': VQ_8, 'AE': AE}