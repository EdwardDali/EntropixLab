import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from enum import Enum
from typing import List, Tuple, Optional, Dict, Union, NamedTuple
import time
from collections import Counter, deque
import math
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device selection with Apple Silicon support
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Set higher precision for matrix multiplication
torch.set_float32_matmul_precision('high')

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E
# Constants for attention masking
MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

class LayerWeights(NamedTuple):
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    ffn_norm: torch.Tensor
    attention_norm: torch.Tensor

class GQAConfig(NamedTuple):
    n_heads: int
    n_kv_heads: int
    head_dim: int

class AttnStats(NamedTuple):
    entropy: torch.Tensor      # Shape: [batch_size, n_layers, num_heads]
    varentropy: torch.Tensor   # Shape: [batch_size, n_layers, num_heads]
    rolling_entropy: deque     # Rolling window of entropy values
    rolling_varentropy: deque  # Rolling window of varentropy values
    window_size: int

    @classmethod
    def new(cls, bsz: int, n_layers: int, n_heads: int, window_size: int = 100) -> 'AttnStats':
        return cls(
            entropy=torch.zeros((bsz, n_layers, n_heads), dtype=torch.float32, device=device),
            varentropy=torch.zeros((bsz, n_layers, n_heads), dtype=torch.float32, device=device),
            rolling_entropy=deque(maxlen=window_size),
            rolling_varentropy=deque(maxlen=window_size),
            window_size=window_size
        )

    def update(self, attention: torch.Tensor, layer_idx: int) -> 'AttnStats':
        """Update statistics with proper dimension handling"""
        attention_probs = F.softmax(attention, dim=-1)
        
        # Calculate entropy per head
        entropy = -torch.sum(
            attention_probs * torch.log2(torch.clamp(attention_probs, min=1e-10)),
            dim=-1
        ).mean(dim=-1)  # Average over sequence length
        
        # Calculate varentropy per head
        mean_entropy = entropy.mean(dim=-1, keepdim=True)
        varentropy = torch.pow(entropy - mean_entropy, 2).mean(dim=-1)
        
        # Update tensors
        self.entropy[:, layer_idx, :] = entropy
        self.varentropy[:, layer_idx, :] = varentropy
        
        # Update rolling statistics
        self.rolling_entropy.append(entropy.mean().item())
        self.rolling_varentropy.append(varentropy.mean().item())
        
        return self

    @property
    def avg_entropy(self) -> float:
        return sum(self.rolling_entropy) / len(self.rolling_entropy) if self.rolling_entropy else 0.0

    @property
    def avg_varentropy(self) -> float:
        return sum(self.rolling_varentropy) / len(self.rolling_varentropy) if self.rolling_varentropy else 0.0

    @property
    def std_error(self) -> float:
        if len(self.rolling_entropy) < 2:
            return 0.0
        return torch.tensor(list(self.rolling_entropy)).std().item() / math.sqrt(len(self.rolling_entropy))

class KVCache:
    """Enhanced Key-Value cache with memory optimization"""
    def __init__(self, max_seq_len: int, n_layers: int, n_heads: int, head_dim: int):
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        # Initialize with smaller chunks and use sparse storage
        chunk_size = 1024  # Smaller initial allocation
        self.k_chunks = [torch.zeros((n_layers, chunk_size, n_heads, head_dim), 
                                   dtype=torch.bfloat16, device=device)]
        self.v_chunks = [torch.zeros((n_layers, chunk_size, n_heads, head_dim), 
                                   dtype=torch.bfloat16, device=device)]
        self.current_pos = 0
        
    def extend_if_needed(self, needed_size: int):
        # Current implementation may lead to memory fragmentation
        # Improved version:
        current_size = sum(chunk.size(1) for chunk in self.k_chunks)
        if current_size >= needed_size:
            return
            
        # Allocate new size with some padding to reduce frequent resizing
        new_size = max(needed_size * 1.5, current_size + 1024)
        new_k_chunk = torch.zeros((self.n_layers, int(new_size - current_size),
                                 self.n_heads, self.head_dim),
                                dtype=torch.bfloat16, device=device)
        new_v_chunk = torch.zeros_like(new_k_chunk)
        self.k_chunks.append(new_k_chunk)
        self.v_chunks.append(new_v_chunk)

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key-value pairs"""
        self.extend_if_needed(pos + k.size(1))
        
        # Find correct chunk and position
        total_pos = 0
        for k_chunk, v_chunk in zip(self.k_chunks, self.v_chunks):
            chunk_size = k_chunk.size(1)
            if total_pos <= pos < total_pos + chunk_size:
                start_idx = pos - total_pos
                end_idx = start_idx + k.size(1)
                k_chunk[layer_idx, start_idx:end_idx] = k
                v_chunk[layer_idx, start_idx:end_idx] = v
                break
            total_pos += chunk_size
        
        # Concatenate all chunks for return
        keys = torch.cat([chunk[layer_idx] for chunk in self.k_chunks], dim=0)[:pos + k.size(1)]
        values = torch.cat([chunk[layer_idx] for chunk in self.v_chunks], dim=0)[:pos + v.size(1)]
        
        return keys, values

    def clear(self):
        """Clear the cache"""
        for chunk in self.k_chunks + self.v_chunks:
            chunk.zero_()
        self.current_pos = 0

class SamplerState(Enum):
    ARGMAX = 0
    SAMPLE = 1
    INSERT_COT = 2
    RESAMPLE = 3
    ADAPTIVE = 4
    EOT = 5

class SamplerConfig:
    def __init__(self, tokenizer=None):
        # Parameter validation methods
        self.temp = self._validate_float("temp", 0.666, min_val=0.1, max_val=2.0)
        self.top_p = self._validate_float("top_p", 0.90, min_val=0.0, max_val=1.0)
        self.top_k = self._validate_int("top_k", 27, min_val=1)
        self.min_p = self._validate_float("min_p", 0.05, min_val=0.0, max_val=1.0)

        # Architecture parameters (will be updated with actual model values)
        self.n_layers = self._validate_int("n_layers", 16, min_val=1)
        self.n_heads = self._validate_int("n_heads", 32, min_val=1)
        self.head_dim = self._validate_int("head_dim", 64, min_val=1)

        # Strategy-specific thresholds
        self.argmax_entropy_thresh = self._validate_float("argmax_entropy_thresh", 0.1, min_val=0.0)
        
        self.sample_min_entropy_thresh = self._validate_float("sample_min_entropy_thresh", 0.1, min_val=0.0)
        self.sample_max_entropy_thresh = self._validate_float("sample_max_entropy_thresh", 1.8, min_val=0.0)
        self.sample_varentropy_thresh = self._validate_float("sample_varentropy_thresh", 0.1, min_val=0.0)
        
        self.cot_min_entropy_thresh = self._validate_float("cot_min_entropy_thresh", 1.8, min_val=0.0)
        self.cot_max_entropy_thresh = self._validate_float("cot_max_entropy_thresh", 2.5, min_val=0.0)
        self.cot_varentropy_thresh = self._validate_float("cot_varentropy_thresh", 0.1, min_val=0.0)
        
        self.resample_min_entropy_thresh = self._validate_float("resample_min_entropy_thresh", 0.5, min_val=0.0)
        self.resample_max_entropy_thresh = self._validate_float("resample_max_entropy_thresh", 2.0, min_val=0.0)
        self.resample_varentropy_thresh = self._validate_float("resample_varentropy_thresh", 3.0, min_val=0.0)
        
        self.adaptive_entropy_thresh = self._validate_float("adaptive_entropy_thresh", 2.5, min_val=0.0)
        self.adaptive_varentropy_thresh = self._validate_float("adaptive_varentropy_thresh", 3.0, min_val=0.0)

        # Enhanced RoPE parameters
        self.max_seq_len = self._validate_int("max_seq_len", 4096, min_val=1)
        self.rope_theta = self._validate_float("rope_theta", 10000.0, min_val=1.0)
        self.rope_scaling = self._validate_float("rope_scaling", 1.0, min_val=0.0)
        self.use_scaled_rope = True
        self.rope_scale_base = self._validate_float("rope_scale_base", 8.0, min_val=1.0)
        self.rope_scale_factor = self._validate_float("rope_scale_factor", 0.25, min_val=0.0)

        # Attention coefficients
        self.helv_attn_ent_offset = self._validate_float("helv_attn_ent_offset", 1.3)
        self.helv_attn_ent_coef = self._validate_float("helv_attn_ent_coef", 0.2)
        self.lehv_interaction_strength_offset = self._validate_float("lehv_interaction_strength_offset", 1.2)
        self.lehv_interaction_strength_coef = self._validate_float("lehv_interaction_strength_coef", 0.3)
        self.hehv_attn_ent_coef = self._validate_float("hehv_attn_ent_coef", 0.2)
        self.hehv_attn_vent_offset = self._validate_float("hehv_attn_vent_offset", 2.0)
        self.hehv_attn_vent_coef = self._validate_float("hehv_attn_vent_coef", 0.5)

        # Enhanced adaptive sampling parameters
        self.n_adaptive_samples = self._validate_int("n_adaptive_samples", 5, min_val=1)
        self.ada_temp_logits = self._validate_float("ada_temp_logits", 0.3)
        self.ada_temp_attn = self._validate_float("ada_temp_attn", 0.2)
        self.ada_temp_agree = self._validate_float("ada_temp_agree", 0.2)
        self.ada_top_p = self._validate_float("ada_top_p", 0.1)
        self.ada_top_k_int = self._validate_float("ada_top_k_int", 0.3)
        self.ada_top_k_agree = self._validate_float("ada_top_k_agree", 0.2)
        self.ada_min_p = self._validate_float("ada_min_p", 0.5)

        # Scoring coefficients
        self.ada_score_logits_ent = self._validate_float("ada_score_logits_ent", 0.1)
        self.ada_score_attn_ent = self._validate_float("ada_score_attn_ent", 0.2)
        self.ada_score_logits_vent = self._validate_float("ada_score_logits_vent", 0.3)
        self.ada_score_attn_vent = self._validate_float("ada_score_attn_vent", 0.4)
        self.ada_score_agree = self._validate_float("ada_score_agree", 0.5)
        self.ada_score_int = self._validate_float("ada_score_int", 0.6)

        # Memory and window parameters
        self.repetition_penalty = self._validate_float("repetition_penalty", 1.2, min_val=1.0)
        self.max_ngram_size = self._validate_int("max_ngram_size", 5, min_val=1)
        self.max_ngram_repeat = self._validate_int("max_ngram_repeat", 3, min_val=1)
        self.strategy_change_batch_size = self._validate_int("strategy_change_batch_size", 1, min_val=1)
        self.window_size = self._validate_int("window_size", 50, min_val=1)
        self.long_window_size = self._validate_int("long_window_size", 500, min_val=1)
        self.decay_factor = self._validate_float("decay_factor", 0.95, min_val=0.0, max_val=1.0)
        self.long_decay_factor = self._validate_float("long_decay_factor", 0.95, min_val=0.0, max_val=1.0)

        # Statistics tracking
        self.stats_window_size = self._validate_int("stats_window_size", 100, min_val=1)
        
        # Special tokens initialization
        if tokenizer:
            try:
                self.initialize_special_tokens(tokenizer)
            except Exception as e:
                logger.warning(f"Could not encode special tokens, using defaults: {str(e)}")
                self.use_default_tokens()
        else:
            self.use_default_tokens()

        self.generator = torch.Generator(device=device).manual_seed(1337)

    def _validate_float(self, name: str, value: float, min_val: float = None, 
                       max_val: float = None) -> float:
        """Validate float parameters"""
        try:
            value = float(value)
            if min_val is not None and value < min_val:
                raise ValueError(f"{name} must be >= {min_val}")
            if max_val is not None and value > max_val:
                raise ValueError(f"{name} must be <= {max_val}")
            return value
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid value for {name}: {value}. {str(e)}")

    def _validate_int(self, name: str, value: int, min_val: int = None,
                     max_val: int = None) -> int:
        """Validate integer parameters"""
        try:
            value = int(value)
            if min_val is not None and value < min_val:
                raise ValueError(f"{name} must be >= {min_val}")
            if max_val is not None and value > max_val:
                raise ValueError(f"{name} must be <= {max_val}")
            return value
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid value for {name}: {value}. {str(e)}")

    def initialize_special_tokens(self, tokenizer):
        """Initialize special tokens from tokenizer"""
        self.cot_token = tokenizer.encode("[COT]", add_special_tokens=False)[0]
        self.eos_token = tokenizer.eos_token_id
        self.bos_token = tokenizer.bos_token_id
        self.pad_token = tokenizer.pad_token_id
        self.stop_tokens = [
            self.eos_token,
            tokenizer.encode("</s>", add_special_tokens=False)[0] if "</s>" in tokenizer.get_vocab() else None,
            tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0] if "<|endoftext|>" in tokenizer.get_vocab() else None
        ]
        self.stop_tokens = [token for token in self.stop_tokens if token is not None]
        logger.info(f"Initialized stop tokens: {self.stop_tokens}")

    def use_default_tokens(self):
        """Use default token values"""
        self.cot_token = 2564
        self.eos_token = 0
        self.bos_token = 1
        self.pad_token = 2
        self.stop_tokens = [0, 2]

class EntropixSampler:
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.strategy_counter = Counter()
        self.recent_tokens = deque(maxlen=200)
        self.current_batch = []
        self.current_strategy = SamplerState.SAMPLE
        self.tokens_since_last_change = 0
        
        # Enhanced metric tracking
        self.entropy_window = deque(maxlen=self.config.window_size)
        self.varentropy_window = deque(maxlen=self.config.window_size)
        self.attention_entropy_window = deque(maxlen=self.config.window_size)
        self.long_entropy_window = deque(maxlen=self.config.long_window_size)
        self.long_varentropy_window = deque(maxlen=self.config.long_window_size)
        
        self.ngram_counts = {}
        self.sliding_window = 100
        
        # Initialize attention stats
        self.attn_stats = None

    @staticmethod
    def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings with improved numerical stability."""
        xq_r = xq[..., ::2]
        xq_i = xq[..., 1::2]
        xk_r = xk[..., ::2]
        xk_i = xk[..., 1::2]
        
        # Reshape freqs_cis for proper broadcasting
        freqs_cis = freqs_cis.view(*([1] * (xq_r.dim() - freqs_cis.dim())), *freqs_cis.shape)
        
        # Complex multiplication with better numerical stability
        xq_out_r = xq_r * freqs_cis.real - xq_i * freqs_cis.imag
        xq_out_i = xq_r * freqs_cis.imag + xq_i * freqs_cis.real
        xk_out_r = xk_r * freqs_cis.real - xk_i * freqs_cis.imag
        xk_out_i = xk_r * freqs_cis.imag + xk_i * freqs_cis.real
        
        # Interleave real and imaginary parts
        xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
        xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)
        
        return xq_out, xk_out

    def apply_scaled_rope(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply scaled rotary positional embeddings"""
        if not self.config.use_scaled_rope:
            return self.apply_rotary_emb(xq, xk, freqs_cis)

        scale = self.config.rope_scale_base ** (self.config.rope_scale_factor * 
                (torch.arange(xq.size(-2), device=device) / xq.size(-2)))
        scaled_freqs = freqs_cis * scale.unsqueeze(-1)
        
        return self.apply_rotary_emb(xq, xk, scaled_freqs)

    @staticmethod
    def build_attention_mask(seqlen: int, start_pos: int, dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
        """Generate an improved causal attention mask with proper scaling."""
        if seqlen <= 1:
            return None
            
        mask = torch.full((seqlen, seqlen), MASK_VALUE, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        
        if start_pos > 0:
            prefix_mask = torch.zeros((seqlen, start_pos), dtype=dtype, device=device)
            mask = torch.cat([prefix_mask, mask], dim=1)
        
        mask = torch.where(mask == MASK_VALUE, mask, torch.zeros_like(mask))
        
        return mask

    @staticmethod
    def grouped_query_attention(
        x: torch.Tensor,
        layer_weights: LayerWeights,
        config: GQAConfig,
        cur_pos: int,
        freqs_cis: torch.Tensor,
        kvcache: KVCache,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
        """Improved attention implementation with grouped-query attention support."""
        bsz, seqlen, _ = x.shape
        
        n_rep = config.n_heads // config.n_kv_heads
        
        xq = F.linear(x, layer_weights.wq)
        xk = F.linear(x, layer_weights.wk)
        xv = F.linear(x, layer_weights.wv)
        
        xq = xq.view(bsz, seqlen, config.n_heads, config.head_dim)
        xk = xk.view(bsz, seqlen, config.n_kv_heads, config.head_dim)
        xv = xv.view(bsz, seqlen, config.n_kv_heads, config.head_dim)
        
        xq, xk = EntropixSampler.apply_rotary_emb(xq, xk, freqs_cis)
        
        keys, values, kvcache = kvcache.update(xk, xv, cur_pos, n_rep)
        
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2).transpose(2, 3)
        values = values.transpose(1, 2)
        
        keys = keys.repeat_interleave(n_rep, dim=1)
        values = values.repeat_interleave(n_rep, dim=1)
        
        scores = torch.matmul(xq, keys) / math.sqrt(config.head_dim)
        scores = scores.to(torch.float32)
        
        if attn_mask is not None:
            scores = scores + attn_mask
        
        scores = torch.where(
            scores <= MASK_VALUE * 0.5,
            torch.full_like(scores, float('-inf')),
            scores
        )
        attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        attention_weights = attention_weights.to(x.dtype)
        
        output = torch.matmul(attention_weights, values)
        output = output.transpose(1, 2).contiguous()
        output = output.view(bsz, seqlen, -1)
        
        output = F.linear(output, layer_weights.wo)
        
        return output, kvcache, scores

    def calculate_metrics(self, logits: torch.Tensor, attention: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics with separate tracking for logits and attention-based metrics"""
        if self.attn_stats is None:
            self.attn_stats = AttnStats.new(
                bsz=attention.size(0),
                n_layers=1,
                n_heads=attention.size(1),
                window_size=self.config.stats_window_size
            )
        
        # Calculate logits-based metrics
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)
        
        # Calculate attention-based metrics
        attn_entropy = self.calculate_attention_entropy(attention)
        attn_varentropy = self.calculate_attention_varentropy(attention)
        agreement = self.calculate_agreement(attention)
        interaction_strength = self.calculate_interaction_strength(attention)
        
        # Update attention stats
        self.attn_stats = self.attn_stats.update(attention, 0)
        
        # Store both raw and rolling metrics
        return {
            "logits_entropy": entropy.mean().item(),
            "logits_varentropy": varentropy.mean().item(),
            "attn_entropy": attn_entropy.mean().item(),
            "attn_varentropy": attn_varentropy.item(),
            "agreement": agreement.item(),
            "interaction_strength": interaction_strength.item(),
            "rolling_entropy": self.attn_stats.avg_entropy,  # Keep for monitoring
            "rolling_varentropy": self.attn_stats.avg_varentropy,  # Keep for monitoring
            "std_error": self.attn_stats.std_error
        }

    def calculate_varentropy_logsoftmax(self, logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate entropy and varentropy using numerically stable methods"""
        log_probs = F.log_softmax(logits, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2
        varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
        return entropy, varentropy

    def calculate_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """Calculate attention entropy with fixed dimension handling"""
        attention_probs = F.softmax(attention, dim=-1)
        entropy = -torch.sum(
            attention_probs * torch.log2(torch.clamp(attention_probs, min=1e-10)),
            dim=-1
        )
        return entropy.mean(dim=1)

    def calculate_attention_varentropy(self, attention: torch.Tensor) -> torch.Tensor:
        """Calculate variance of attention entropy with fixed dimension handling"""
        entropy = self.calculate_attention_entropy(attention)
        varentropy = torch.var(entropy, dim=-1)
        varentropy = torch.where(torch.isnan(varentropy), torch.zeros_like(varentropy), varentropy)
        return varentropy.mean()

    def calculate_agreement(self, attention: torch.Tensor) -> torch.Tensor:
        """Calculate agreement between attention heads"""
        attention_probs = F.softmax(attention, dim=-1)
        mean_attention = attention_probs.mean(dim=1, keepdim=True)
        head_agreement = 1 - torch.abs(attention_probs - mean_attention).mean()
        return head_agreement

    def calculate_interaction_strength(self, attention: torch.Tensor) -> torch.Tensor:
        """Calculate interaction strength between attention heads"""
        attention_probs = F.softmax(attention, dim=-1)
        batch_size, num_heads, seq_len, _ = attention_probs.shape
        attention_flat = attention_probs.view(batch_size, num_heads, -1)
        norm = torch.norm(attention_flat, dim=2, keepdim=True)
        normalized = attention_flat / (norm + 1e-8)
        interaction = torch.matmul(normalized, normalized.transpose(-1, -2))
        return torch.mean(torch.abs(interaction))

    def _sample(
        self, 
        logits: torch.Tensor, 
        temperature: float = None, 
        top_p: float = None, 
        top_k: int = None, 
        min_p: float = None
    ) -> torch.Tensor:
        """
        Enhanced sampling with multiple controls.
        
        Args:
            logits: Model logits
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            min_p: Minimum probability threshold
            
        Returns:
            torch.Tensor: Selected token
        """
        try:
            temperature = temperature or self.config.temp
            top_p = top_p or self.config.top_p
            top_k = top_k or self.config.top_k
            min_p = min_p or self.config.min_p

            if logits.dim() == 3:
                logits = logits[:, -1, :]
            elif logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            # Apply repetition penalty
            if len(self.recent_tokens) > 0:
                for token in set(self.recent_tokens):
                    logits[:, token] = logits[:, token] / self.config.repetition_penalty

            logits = logits / temperature

            # Apply min-p filtering
            if min_p > 0.0:
                sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > (1 - min_p)
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1,
                    index=torch.argsort(logits, descending=True),
                    src=sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Apply top-k sampling
            top_k = min(top_k, logits.size(-1))
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
                
                # Apply top-p sampling to top-k candidates
                cumulative_probs = torch.cumsum(F.softmax(top_k_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                top_k_logits = top_k_logits.masked_fill(sorted_indices_to_remove, float('-inf'))

                probs = F.softmax(top_k_logits, dim=-1)
                sample_idx = torch.multinomial(probs, num_samples=1, generator=self.config.generator)
                
                sampled_token = torch.gather(top_k_indices, -1, sample_idx)
            else:
                probs = F.softmax(logits, dim=-1)
                sampled_token = torch.multinomial(probs, num_samples=1, generator=self.config.generator)

            return sampled_token.to(torch.int32)
            
        except Exception as e:
            logger.error(f"Error in sampling: {str(e)}")
            # Emergency fallback: return most likely token
            return torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)

    def _adaptive_sample(self, logits: torch.Tensor, metrics: Dict[str, float]) -> torch.Tensor:
        """
        Perform adaptive sampling based on current metrics.
        
        Args:
            logits: Model logits
            metrics: Dictionary of current metrics
            
        Returns:
            torch.Tensor: Selected token
        """
        try:
            # Calculate uncertainty metrics
            logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
            attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]
            
            # Adjust temperature based on uncertainty and agreement
            temperature = self.config.temp * (
                1 + self.config.ada_temp_logits * logits_uncertainty +
                self.config.ada_temp_attn * attn_uncertainty -
                self.config.ada_temp_agree * metrics["agreement"]
            )
            temperature = max(0.1, min(2.0, temperature))
            
            # Adjust top-p based on attention variance
            top_p = min(max(
                self.config.top_p * (1 + self.config.ada_top_p * metrics["attn_varentropy"]),
                0.1
            ), 1.0)
            
            # Adjust top-k based on interaction strength and agreement
            top_k = int(min(max(
                self.config.top_k * (
                    1 + self.config.ada_top_k_int * metrics["interaction_strength"] -
                    self.config.ada_top_k_agree * metrics["agreement"]
                ),
                5
            ), 100))
            
            # Adjust min-p based on uncertainty
            min_p = min(max(
                self.config.min_p * (1 - self.config.ada_min_p * logits_uncertainty),
                0.01
            ), 0.5)
            
            logger.debug(
                f"Adaptive parameters:\n"
                f"  Temperature: {temperature:.3f}\n"
                f"  Top-p: {top_p:.3f}\n"
                f"  Top-k: {top_k}\n"
                f"  Min-p: {min_p:.3f}"
            )
            
            # Generate multiple samples
            samples = []
            for _ in range(self.config.n_adaptive_samples):
                sample = self._sample(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p
                )
                samples.append(sample)
            
            # Score each sample and select the best one
            sample_scores = []
            for sample in samples:
                score = self.score_sample(sample, logits, metrics)
                sample_scores.append(score)
                logger.debug(f"Sample score: {score:.3f}")
            
            best_sample_idx = torch.argmax(torch.tensor(sample_scores))
            logger.debug(f"Selected sample index: {best_sample_idx}")
            
            return samples[best_sample_idx]
            
        except Exception as e:
            logger.error(f"Error in adaptive sampling: {str(e)}")
            logger.info("Falling back to basic sampling")
            return self._sample(logits, self.config.temp)

    def determine_strategy(self, entropy: float, varentropy: float, attention_entropy: float) -> SamplerState:
        """Enhanced strategy determination using configurable thresholds"""
        recent_tokens = list(self.recent_tokens)[-1:] if self.recent_tokens else []
        if recent_tokens and self.config.stop_tokens and any(token in self.config.stop_tokens for token in recent_tokens):
            return SamplerState.EOT

        # Log current values and thresholds for debugging
        logger.debug(f"\nCurrent Values:")
        logger.debug(f"Entropy: {entropy:.4f}")
        logger.debug(f"Varentropy: {varentropy:.4f}")
        logger.debug(f"Attention Entropy: {attention_entropy:.4f}")

        # Check for ARGMAX strategy (low entropy, low variance)
        if entropy < self.config.argmax_entropy_thresh and varentropy < self.config.sample_varentropy_thresh:
            logger.debug("ARGMAX triggered (low entropy, low variance)")
            return SamplerState.ARGMAX

        # Check for INSERT_COT strategy (high entropy, low variance)
        if (self.config.cot_min_entropy_thresh <= entropy <= self.config.cot_max_entropy_thresh and 
            varentropy < self.config.cot_varentropy_thresh):
            logger.debug("INSERT_COT triggered (high entropy, low variance)")
            return SamplerState.INSERT_COT

        # Check for RESAMPLE strategy (high entropy, high variance)
        if (entropy >= self.config.resample_min_entropy_thresh and 
            varentropy >= self.config.resample_varentropy_thresh):
            logger.debug("RESAMPLE triggered (high entropy, high variance)")
            return SamplerState.RESAMPLE

        # Check for SAMPLE strategy (low entropy, high variance)
        if (entropy <= self.config.sample_max_entropy_thresh and 
            varentropy >= self.config.sample_varentropy_thresh):
            logger.debug("SAMPLE triggered (low entropy, high variance)")
            return SamplerState.SAMPLE

        # Check for ADAPTIVE strategy (near center point)
        center_distance = math.sqrt(
            (entropy - self.config.adaptive_entropy_thresh) ** 2 + 
            (varentropy - self.config.adaptive_varentropy_thresh) ** 2
        )
        
        if center_distance < 1.5:  # Using adaptive radius for center zone
            logger.debug("ADAPTIVE triggered (central position)")
            return SamplerState.ADAPTIVE

        # Default to SAMPLE if no other strategy matches
        logger.debug("SAMPLE triggered (default)")
        return SamplerState.SAMPLE

    def score_sample(self, sample: torch.Tensor, logits: torch.Tensor, metrics: Dict[str, float]) -> float:
        """
        Score a sample based on log probability and metrics.
        
        Args:
            sample: Generated token
            logits: Model logits
            metrics: Dictionary of current metrics
            
        Returns:
            float: Combined score for the sample
        """
        try:
            # Calculate log probability score
            sample_flat = sample.flatten().to(torch.long)
            one_hot = F.one_hot(sample_flat, logits.shape[-1])
            log_probs = F.log_softmax(logits, dim=-1).view(-1, logits.shape[-1])
            log_prob = torch.sum(log_probs * one_hot).item()
            
            # Calculate confidence score based on multiple metrics
            confidence_score = (
                (1 - metrics["logits_entropy"]) * self.config.ada_score_logits_ent +
                (1 - metrics["attn_entropy"]) * self.config.ada_score_attn_ent +
                (1 - metrics["logits_varentropy"]) * self.config.ada_score_logits_vent +
                (1 - metrics["attn_varentropy"]) * self.config.ada_score_attn_vent +
                metrics["agreement"] * self.config.ada_score_agree +
                metrics["interaction_strength"] * self.config.ada_score_int
            )
            
            return log_prob + confidence_score
            
        except Exception as e:
            logger.error(f"Error in score_sample: {str(e)}")
            return float('-inf')
    
    def sample(self, logits: torch.Tensor, attention: torch.Tensor) -> Tuple[torch.Tensor, SamplerState]:
        """Main sampling method"""
        if not isinstance(logits, torch.Tensor) or not isinstance(attention, torch.Tensor):
            raise TypeError("Inputs must be torch.Tensor objects")
        
        if logits.dim() != 3:
            raise ValueError(f"Expected 3D logits tensor, got shape {logits.shape}")
        
        batch_size, seq_len, vocab_size = logits.shape
        attention_shape = attention.shape
        
        logger.debug(f"Logits shape: {logits.shape}")
        logger.debug(f"Attention shape: {attention_shape}")
        
        if len(attention_shape) != 4:
            raise ValueError(f"Expected 4D attention tensor, got shape {attention_shape}")
        
        metrics = self.calculate_metrics(logits, attention)
        
        self.entropy_window.append(metrics["logits_entropy"])
        self.varentropy_window.append(metrics["logits_varentropy"])
        self.attention_entropy_window.append(metrics["attn_entropy"])
        self.long_entropy_window.append(metrics["logits_entropy"])
        self.long_varentropy_window.append(metrics["logits_varentropy"])
        
        avg_entropy = self.weighted_average(list(self.entropy_window), self.config.decay_factor)
        avg_varentropy = self.weighted_average(list(self.varentropy_window), self.config.decay_factor)
        avg_attention_entropy = self.weighted_average(list(self.attention_entropy_window), self.config.decay_factor)
        
        long_avg_entropy = self.weighted_average(list(self.long_entropy_window), self.config.long_decay_factor)
        long_avg_varentropy = self.weighted_average(list(self.long_varentropy_window), self.config.long_decay_factor)
        
        combined_entropy = (avg_entropy + long_avg_entropy) / 2
        combined_varentropy = (avg_varentropy + long_avg_varentropy) / 2
        
        self.current_strategy = self.determine_strategy(
            combined_entropy,
            combined_varentropy,
            avg_attention_entropy
        )
        
        if self.current_strategy == SamplerState.ARGMAX:
            sampled_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        
        elif self.current_strategy == SamplerState.INSERT_COT:
            if self.config.cot_token not in self.recent_tokens:
                sampled_token = torch.tensor([[self.config.cot_token]], device=device)
            else:
                temp_adj = (self.config.helv_attn_ent_offset + 
                            self.config.helv_attn_ent_coef * avg_attention_entropy)
                sampled_token = self._sample(logits, temperature=min(1.5, self.config.temp * temp_adj))
        
        elif self.current_strategy == SamplerState.RESAMPLE:
            temp_adj = (self.config.lehv_interaction_strength_offset + 
                        self.config.lehv_interaction_strength_coef * metrics["interaction_strength"])
            top_k_adj = max(5, int(self.config.top_k * (1 + 0.5 * (1 - metrics["agreement"]))))
            sampled_token = self._sample(
                logits,
                temperature=min(1.5, self.config.temp * temp_adj),
                top_k=top_k_adj
            )
        
        elif self.current_strategy == SamplerState.ADAPTIVE:
            sampled_token = self._adaptive_sample(logits, metrics)
        
        elif self.current_strategy == SamplerState.EOT:
            sampled_token = torch.tensor([[self.config.eos_token]], device=device)
        
        else:  # SamplerState.SAMPLE
            sampled_token = self._sample(logits)
        
        self.strategy_counter[self.current_strategy.name] += 1
        self.tokens_since_last_change += 1
        self.current_batch.append(sampled_token.item())
        self.recent_tokens.append(sampled_token.item())
        
        if self.check_ngram_repetition(list(self.recent_tokens)):
            sampled_token = self._sample(
                logits,
                temperature=1.2,
                top_k=100
            )
        
        if len(self.current_batch) >= self.config.strategy_change_batch_size:
            self.current_batch = []
            self.tokens_since_last_change = 0
        
        return sampled_token, self.current_strategy

    def check_ngram_repetition(self, tokens: List[int]) -> bool:
        """Check for repeated n-grams in recent tokens"""
        if len(tokens) < self.config.max_ngram_size:
            return False
            
        window = tokens[-self.sliding_window:]
        
        if len(self.ngram_counts) > 10000:
            self.ngram_counts = {}
        
        for n in range(2, min(self.config.max_ngram_size + 1, len(window))):
            ngrams = [tuple(window[i:i+n]) for i in range(len(window)-n+1)]
            
            for ngram in ngrams:
                if ngram in self.ngram_counts:
                    self.ngram_counts[ngram] += 1
                    if self.ngram_counts[ngram] > self.config.max_ngram_repeat:
                        return True
                else:
                    self.ngram_counts[ngram] = 1
                    
        return False

    def reset_ngram_counts(self):
        """Reset the n-gram counter"""
        self.ngram_counts = {}

    def weighted_average(self, values: List[float], decay_factor: float) -> float:
        """Calculate weighted average with exponential decay"""
        if not values:
            return 0.0
            
        weights = [decay_factor ** i for i in range(len(values) - 1, -1, -1)]
        weighted_sum = sum(w * v for w, v in zip(weights, values))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 1000) -> str:
    """Generate response using the enhanced sampling strategy with fixed dimension handling"""
    # Get model architecture parameters
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    
    # Initialize config with model-specific parameters
    cfg = SamplerConfig(tokenizer)
    cfg.head_dim = head_dim
    cfg.n_heads = n_heads
    cfg.n_layers = n_layers
    
    sampler = EntropixSampler(cfg)
    
    input_ids = tokenizer.encode(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=model.config.max_position_embeddings - max_tokens
    ).to(device)
    
    attention_mask = torch.ones_like(input_ids)
    
    generated_text = ""
    start_time = time.time()
    
    logger.info(f"Generating response for prompt: '{prompt}'")
    logger.info(f"Model architecture: {n_layers} layers, {n_heads} heads, {head_dim} head dimension")
    print("Generated text:", flush=True)
    
    with torch.inference_mode():
        for i in range(max_tokens):
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                output_attentions=True
            )
            
            logits = outputs.logits
            attention_layers = outputs.attentions
            last_layer_attention = attention_layers[-1].to(device)
            
            try:
                sampled_token, state = sampler.sample(logits, last_layer_attention)
            except Exception as e:
                logger.error(f"Error in sampling: {str(e)}")
                logger.error(f"Attention shape: {last_layer_attention.shape}")
                logger.error(f"Logits shape: {logits.shape}")
                raise
            
            if state == SamplerState.EOT or sampled_token[0] == tokenizer.eos_token_id:
                break
            
            # Updated COT insertion with more descriptive marker
            if state == SamplerState.INSERT_COT and sampled_token[0] == cfg.cot_token:
                next_token_text = "🤔[Let me think about this]"
                print("\n" + next_token_text + "\n", end="", flush=True)
            else:
                next_token_text = tokenizer.decode(sampled_token[0])
                print(next_token_text, end="", flush=True)
                
            generated_text += next_token_text
            
            input_ids = torch.cat([input_ids, sampled_token.transpose(0, 1)], dim=-1)
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((1, 1), dtype=torch.long, device=device)
            ], dim=1)
            
            if input_ids.shape[1] >= model.config.max_position_embeddings:
                logger.warning("Reached maximum sequence length. Stopping generation.")
                break
    
    total_time = time.time() - start_time
    logger.info(f"Generation completed in {total_time:.2f} seconds")
    
    # Log statistics
    total_tokens = sum(sampler.strategy_counter.values())
    logger.info("\nToken Generation Strategy Distribution:")
    for strategy, count in sampler.strategy_counter.items():
        percentage = (count / total_tokens) * 100
        logger.info(f"{strategy}: {count} ({percentage:.2f}%)")
    
    return generated_text

# Add this helper function to diagnose attention shape issues
def debug_attention_shape(attention_tensor: torch.Tensor, name: str = "attention"):
    """Helper function to debug attention tensor shapes"""
    logger.info(f"{name} tensor shape: {attention_tensor.shape}")
    logger.info(f"{name} tensor device: {attention_tensor.device}")
    if torch.isnan(attention_tensor).any():
        logger.warning(f"Found NaN values in {name} tensor!")
    logger.info(f"{name} tensor stats - min: {attention_tensor.min():.4f}, max: {attention_tensor.max():.4f}, "
                f"mean: {attention_tensor.mean():.4f}, std: {attention_tensor.std():.4f}")

def main():
    """Main function to run the enhanced text generation"""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            attn_implementation="eager"
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    print("Type 'quit' to exit the program.")
    
    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ").strip()
        if prompt.lower() == 'quit':
            break
        if not prompt:
            print("Please enter a non-empty prompt.")
            continue
        
        try:
            response = generate_response(model, tokenizer, prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Generated response: {response}")
            print("\n" + "-"*50 + "\n")
            
            with open("generated_response.txt", "w", encoding="utf-8") as file:
                file.write(f"Prompt: {prompt}\n\nGenerated response: {response}")
            print("Response saved to generated_response.txt")
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
    
    print("Thank you for using the AI assistant. Goodbye!")

if __name__ == "__main__":
    main()