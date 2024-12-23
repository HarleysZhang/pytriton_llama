import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .model_config import Qwen2Config
from .RotaryEmbedding import Qwen2RotaryEmbedding
from ..kernels import *


class Attention(nn.Module):
    def __init__(self, num_heads_q: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads_q = num_heads_q
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads_q * head_dim
        
    def context_forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index:int,
        qk_scale = None,
    ) -> torch.Tensor:
        xq = xq.to(torch.float16)
        batch_size, seq_len, _, _ = xq.shape  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)
        
        # 1. 获取 prefill 阶段的 cur_select_index, 并更新 kv cache 张量
        combined_kv = torch.cat([xk, xv], dim=2) # (B, L, 2*num_kv_heads, head_dim)
        combined_kv_reshaped = combined_kv.view(-1, self.num_kv_heads*2, self.head_dim)
        # 更新 kv_buffer, atten_info.kv_buffer[layer_index]
        update_kv_buffer(combined_kv_reshaped, atten_info.cur_select_index, atten_info.kv_buffer[layer_index])

        # 2. sel-attention. flashattention 计算: softmax(qk^t) * v
        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)
        output = flash_attention_v2(xq, keys, values, qk_scale)
        
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size))
        
        return output

    def token_forward(self, 
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        atten_info,
        layer_index:int,
        qk_scale = None, # 计算 attention 分数缩放的系数
    ) -> torch.Tensor:
        xq = xq.to(torch.float16)
        batch_size, seq_len, num_heads_q, _ = xq.shape  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)

        # 1. 先获取 kv 缓冲向量再更新 kv 向量
        xq = xq.view(batch_size, num_heads_q, self.head_dim)
        combined_kv = torch.cat([xk, xv], dim=-2) # (B, L, 2*num_kv_heads, head_dim)
        reshaped_kv = combined_kv.view(-1, 2 * self.num_kv_heads, self.head_dim)
        
        # 更新 kv_buffer, atten_info.kv_buffer[layer_index]
        update_kv_buffer(reshaped_kv, atten_info.cur_select_index, atten_info.kv_buffer[layer_index])

        # 2. flashattention 计算: softmax(qk^t) * v
        output = flash_decoding(
            xq,
            atten_info.kv_buffer[layer_index][:, : self.num_kv_heads, :], 
            atten_info.kv_buffer[layer_index][:, self.num_kv_heads:, :], 
            qk_scale,
            atten_info.start_index, 
            atten_info.b_seq_len, 
            atten_info.max_actual_seq_len
        ) # ouput shape is [batchs, num_heads, head_dim]

        output = output.view(batch_size, seq_len, self.hidden_size) # 输出张量 seq_len = 1

        return output

class Qwen2Attention(nn.Module):
    def __init__(self,  
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # K V 头数相同，但和 Q 可能不同
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj_weight = nn.Parameter(torch.rand(hidden_size, hidden_size, dtype=torch.float16))
        self.q_proj_bias = nn.Parameter(torch.rand(hidden_size, dtype=torch.float16))

        self.kv_proj_weight = nn.Parameter(torch.rand(self.num_kv_heads * self.head_dim * 2, self.hidden_size, dtype=torch.float16))
        self.kv_proj_bias = nn.Parameter(torch.rand(self.num_kv_heads * self.head_dim * 2, dtype=torch.float16))
        self.o_proj_weight = nn.Parameter(torch.rand(hidden_size, hidden_size, dtype=torch.float16))

        self.attn = Attention(num_heads, num_kv_heads, self.head_dim)

    def _get_qkv(
        self, 
        x: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape  # prefill: (B, Seq_Len, Dim); decode: (B, 1, Dim)
        
        xq = F.linear(x, self.q_proj_weight.data, bias=self.q_proj_bias.data)
        xkv = F.linear(x, self.kv_proj_weight.data, bias=self.kv_proj_bias.data)
        xk, xv = torch.split(xkv, self.num_kv_heads * self.head_dim, dim=-1)

        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = rope_forward(xq, xk, cos, sin)

        return xq, xk, xv
    
    def forward(
        self,
        x: torch.Tensor,
        atten_info,
        layer_index:int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        qk_scale = None,
    ) -> torch.Tensor:
        _, seq_len, _ = x.shape

        # 计算 attention 的输入 q、k、v
        xq, xk, xv = self._get_qkv(x, position_embeddings)

        # 根据输入张量 seq_len 长度选择 context_forward 还是 token_forward
        if seq_len > 1:
            attn_output = self.attn.context_forward(
                xq, xk, xv,
                atten_info, layer_index,
                qk_scale,
            )
            # if torch.isnan(attn_output).any(): # 检查 NaNs
            #     raise ValueError(f"NaNs detected in context_forward output at layer {layer_index}")    
        else:
            attn_output = self.attn.token_forward(
                xq, xk, xv, 
                atten_info, layer_index,
                qk_scale,
            )
            # if torch.isnan(attn_output).any(): # 检查 NaNs
            #     raise ValueError(f"NaNs detected in token_forward output at layer {layer_index}")    

        # 进行张量矩阵乘法, 需要对原始的 o_proj_weight 权重进行转置, attn_output shape is [batch_size, seq_len, hidden_size]
        output = F.linear(attn_output, self.o_proj_weight.data)
        return output
    
class FusedMLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=torch.float16) # torch.float32 cpu

    def forward(self, x):
        return self.down_proj(swiglu_forward(self.gate_proj(x), self.up_proj(x)))
        
class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config= config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim if config.head_dim is not None else config.hidden_size // config.num_heads

        self.rmsnorm_eps = config.rms_norm_eps

        # 命名和 Qwen2ForCausalLM 一致
        self.input_layernorm_weight = nn.Parameter(torch.ones(self.hidden_size, dtype=torch.float16))
        self.post_attention_layernorm_weight = nn.Parameter(torch.ones(self.hidden_size, dtype=torch.float16))
        
        self.self_attn = Qwen2Attention(self.hidden_size, self.num_heads, self.num_kv_heads)
        self.mlp = FusedMLP(config)

    def forward(self, 
        hidden_states: torch.Tensor, 
        atten_info,
        layer_index: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        qk_scale = None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states, residual = skip_rmsnorm(hidden_states, residual, self.input_layernorm_weight.data, self.rmsnorm_eps)

        # 调用 attention 模块
        hidden_states = self.self_attn(hidden_states, atten_info, layer_index, position_embeddings, qk_scale)
        
        # 调用 mlp 模块
        hidden_states, residual = skip_rmsnorm(hidden_states, residual, self.post_attention_layernorm_weight.data, self.rmsnorm_eps)
        hidden_states = self.mlp.forward(hidden_states) # 调用 Feed Forward 模块并做残差连接
        
        return hidden_states, residual

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()

        assert config.vocab_size != -1, "Vocab size must be set"
        self.rmsnorm_eps = config.rms_norm_eps

        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_layers = config.num_layers
        head_dim = config.head_dim if config.head_dim is not None else config.hidden_size // config.num_heads

        self.qk_scale = 1.0 / (head_dim ** 0.5)

        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        
        # Embedding 层权重的形状为 (vocab_size, hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.float16)
        self.norm_weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))

        # 使用 nn.Linear 层替代 lm_head_weight
        self.lm_head_weight = nn.Parameter(torch.rand(vocab_size, hidden_size, dtype=torch.float16))
        
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config) for _ in range(num_layers)]
        )

        # self.hidden_states = []

    def forward(
        self, input_ids: torch.Tensor, start_pos, atten_info,
        position_ids: torch.Tensor = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        # self.hidden_states = []
        batch_size, seq_len = input_ids.shape
        residual = None

        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.get_input_embeddings(input_ids)

        if position_ids is None:
            cache_position = torch.arange(start_pos, start_pos + seq_len, device=h.device)
            position_ids = cache_position.unsqueeze(0)

        if seq_len > 1:
            qk_scale = self.qk_scale * 1.4426950408889634
        else:
            qk_scale = self.qk_scale
            position_ids = position_ids.repeat(batch_size, 1)

        position_embeddings = self.rotary_emb(h, position_ids)
       
        # Consecutively apply all the encoder layers
        for i, layer in enumerate(self.layers):            
            # self.hidden_states.append(h)
            h, residual = layer(h, atten_info, i, position_embeddings, qk_scale, residual)  # h.shape [batch_size, seq_len, hidden_dim]

        h, _ = skip_rmsnorm(h, residual, self.norm_weight.data, self.rmsnorm_eps)        
        output = F.linear(h, self.lm_head_weight.data)
        return output
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    