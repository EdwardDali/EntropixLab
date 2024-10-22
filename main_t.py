import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from enum import Enum
from typing import List, Tuple, Optional, Dict, Union
import time
from collections import Counter, deque
import math

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

class SamplerState(Enum):
    ARGMAX = 0
    SAMPLE = 1
    INSERT_COT = 2
    RESAMPLE = 3
    ADAPTIVE = 4
    EOT = 5  # New state for end of text handling

class SamplerConfig:
    def __init__(self, tokenizer=None):
        # Sampling parameters
        self.temp = 0.666
        self.top_p = 0.90
        self.top_k = 27
        self.min_p = 0.03

        # Entropy thresholds
        self.low_ent_thresh = 0.1
        self.low_vent_thresh = 0.1
        self.med_ent_thresh = 3.0
        self.high_ent_thresh = 5.0
        self.high_vent_thresh = 5.0
        self.varentropy_threshold = 0.1

        # Position encoding parameters
        self.max_seq_len = 4096
        self.head_dim = 64
        self.rope_theta = 10000.0
        self.rope_scaling = 1.0
        
        # Attention coefficients
        self.helv_attn_ent_offset = 1.3
        self.helv_attn_ent_coef = 0.2
        self.lehv_interaction_strength_offset = 1.2
        self.lehv_interaction_strength_coef = 0.3
        self.hehv_attn_ent_coef = 0.2
        self.hehv_attn_vent_offset = 2.0
        self.hehv_attn_vent_coef = 0.5

        # Adaptive sampling parameters
        self.n_adaptive_samples = 5
        self.ada_temp_logits = 0.3
        self.ada_temp_attn = 0.2
        self.ada_temp_agree = 0.2
        self.ada_top_p = 0.1
        self.ada_top_k_int = 0.3
        self.ada_top_k_agree = 0.2
        self.ada_min_p = 0.5
        self.ada_score_logits_ent = 0.1
        self.ada_score_attn_ent = 0.2
        self.ada_score_logits_vent = 0.3
        self.ada_score_attn_vent = 0.4
        self.ada_score_agree = 0.5
        self.ada_score_int = 0.6

        # Special tokens and generation control
        if tokenizer:
            try:
                self.cot_token = tokenizer.encode("[COT]", add_special_tokens=False)[0]
                self.eos_token = tokenizer.eos_token_id
                self.bos_token = tokenizer.bos_token_id
                self.pad_token = tokenizer.pad_token_id
                # Initialize stop tokens list with known special tokens
                self.stop_tokens = [
                    self.eos_token,
                    tokenizer.encode("</s>", add_special_tokens=False)[0] if "</s>" in tokenizer.get_vocab() else None,
                    tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0] if "<|endoftext|>" in tokenizer.get_vocab() else None
                ]
                # Remove None values from stop_tokens
                self.stop_tokens = [token for token in self.stop_tokens if token is not None]
                logger.info(f"Initialized stop tokens: {self.stop_tokens}")
            except Exception as e:
                logger.warning(f"Could not encode special tokens, using defaults: {str(e)}")
                self.cot_token = 2564
                self.eos_token = 0
                self.bos_token = 1
                self.pad_token = 2
                self.stop_tokens = [0, 2]  # Default stop tokens
        else:
            # Default values if no tokenizer is provided
            self.cot_token = 2564
            self.eos_token = 0
            self.bos_token = 1
            self.pad_token = 2
            self.stop_tokens = [0, 2]

        # Memory and window parameters
        self.repetition_penalty = 1.2
        self.max_ngram_size = 5
        self.max_ngram_repeat = 3
        self.strategy_change_batch_size = 1
        self.window_size = 50
        self.long_window_size = 500
        self.decay_factor = 0.95
        self.long_decay_factor = 0.95

        self.generator = torch.Generator(device=device).manual_seed(1337)

class EntropixSampler:
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.strategy_counter = Counter()
        self.recent_tokens = deque(maxlen=200)
        self.current_batch = []
        self.current_strategy = SamplerState.SAMPLE
        self.tokens_since_last_change = 0
        
        # Metric tracking windows
        self.entropy_window = deque(maxlen=self.config.window_size)
        self.varentropy_window = deque(maxlen=self.config.window_size)
        self.attention_entropy_window = deque(maxlen=self.config.window_size)
        self.long_entropy_window = deque(maxlen=self.config.long_window_size)
        self.long_varentropy_window = deque(maxlen=self.config.long_window_size)
        
        self.ngram_counts = {}
        self.sliding_window = 100

    def build_attention_mask(self, seqlen: int, start_pos: int) -> Optional[torch.Tensor]:
        """Generate causal attention mask"""
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=device), mask])
            return mask.to(torch.float32)
        return None

    def precompute_rope_freqs(self, dim: int, end: int, theta: float = 10000.0, scale: float = 1.0) -> torch.Tensor:
        """Precompute RoPE frequency bands"""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[:(dim // 2)] / dim))
        freqs = freqs * scale  # Apply scaling
        t = torch.arange(end, device=device).unsqueeze(1)
        freqs = freqs.unsqueeze(0)
        freqs = t * freqs
        return torch.exp(1j * freqs)

    def calculate_varentropy_logsoftmax(self, logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate entropy and varentropy using log_softmax for numerical stability"""
        log_probs = F.log_softmax(logits, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2
        varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
        return entropy, varentropy

    def calculate_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """Calculate attention entropy across heads"""
        attention_probs = F.softmax(attention, dim=-1)
        entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, min=1e-10)), dim=-1)
        return entropy.mean(dim=1)

    def calculate_attention_varentropy(self, attention: torch.Tensor) -> torch.Tensor:
        """Calculate variance of attention entropy"""
        attention_entropy = self.calculate_attention_entropy(attention)
        varentropy = torch.var(attention_entropy)
        return torch.where(torch.isnan(varentropy), torch.zeros_like(varentropy), varentropy)

    def calculate_agreement(self, attention: torch.Tensor) -> torch.Tensor:
        """Calculate agreement between attention heads"""
        attention_probs = F.softmax(attention, dim=-1)
        mean_attention = torch.mean(attention_probs, dim=1, keepdim=True)
        return 1 - torch.mean(torch.abs(attention_probs - mean_attention))

    def calculate_interaction_strength(self, attention: torch.Tensor) -> torch.Tensor:
        """Calculate interaction strength from attention scores"""
        return torch.mean(torch.abs(attention))

    def calculate_metrics(self, logits: torch.Tensor, attention: torch.Tensor) -> Dict[str, float]:
        """Calculate all sampling metrics"""
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)
        attn_entropy = self.calculate_attention_entropy(attention)
        attn_varentropy = self.calculate_attention_varentropy(attention)
        agreement = self.calculate_agreement(attention)
        interaction_strength = self.calculate_interaction_strength(attention)

        return {
            "logits_entropy": entropy.mean().item(),
            "logits_varentropy": varentropy.mean().item(),
            "attn_entropy": attn_entropy.mean().item(),
            "attn_varentropy": attn_varentropy.item(),
            "agreement": agreement.item(),
            "interaction_strength": interaction_strength.item(),
            "logits_uncertainty": entropy.mean().item() + varentropy.mean().item(),
            "attn_uncertainty": attn_entropy.mean().item() + attn_varentropy.item()
        }

    def _sample(
        self, 
        logits: torch.Tensor, 
        temperature: float = None, 
        top_p: float = None, 
        top_k: int = None, 
        min_p: float = None
    ) -> torch.Tensor:
        """Enhanced sampling with temperature, top-p, top-k, and min-p controls"""
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

        # Apply min-p sampling
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
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
        
        # Apply top-p sampling
        cumulative_probs = torch.cumsum(F.softmax(top_k_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        top_k_logits = top_k_logits.masked_fill(sorted_indices_to_remove, float('-inf'))

        probs = F.softmax(top_k_logits, dim=-1)
        sample = torch.multinomial(probs, num_samples=1, generator=self.config.generator)
        return top_k_indices.gather(-1, sample)

    def _adaptive_sample(self, logits: torch.Tensor, metrics: Dict[str, float]) -> torch.Tensor:
        """Adaptive sampling based on current metrics"""
        # Adjust parameters based on metrics
        temperature = self.config.temp * (
            1 + self.config.ada_temp_logits * metrics["logits_uncertainty"] + 
            self.config.ada_temp_attn * metrics["attn_uncertainty"] - 
            self.config.ada_temp_agree * metrics["agreement"]
        )
        
        top_p = min(max(
            self.config.top_p * (1 + self.config.ada_top_p * metrics["attn_varentropy"]), 
            0.1
        ), 1.0)
        
        top_k = int(min(max(
            self.config.top_k * (1 + self.config.ada_top_k_int * metrics["interaction_strength"] - 
            self.config.ada_top_k_agree * metrics["agreement"]), 
            1
        ), 100))
        
        min_p = min(max(
            self.config.min_p * (1 - self.config.ada_min_p * metrics["logits_uncertainty"]), 
            0.01
        ), 0.5)

        # Generate multiple samples and select best
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

        sample_scores = [self.score_sample(sample, logits, metrics) for sample in samples]
        best_sample_idx = torch.argmax(torch.tensor(sample_scores))
        return samples[best_sample_idx]

    def score_sample(self, sample: torch.Tensor, logits: torch.Tensor, metrics: Dict[str, float]) -> float:
        """Score a sample based on log probability and metrics"""
        sample_flat = sample.flatten().to(torch.long)
        one_hot = F.one_hot(sample_flat, logits.shape[-1])
        log_probs = F.log_softmax(logits, dim=-1).view(-1, logits.shape[-1])
        log_prob = torch.sum(log_probs * one_hot).item()
        
        confidence_score = (
            (1 - metrics["logits_entropy"]) * self.config.ada_score_logits_ent +
            (1 - metrics["attn_entropy"]) * self.config.ada_score_attn_ent +
            (1 - metrics["logits_varentropy"]) * self.config.ada_score_logits_vent +
            (1 - metrics["attn_varentropy"]) * self.config.ada_score_attn_vent +
            metrics["agreement"] * self.config.ada_score_agree +
            metrics["interaction_strength"] * self.config.ada_score_int
        )
        return log_prob + confidence_score

    def weighted_average(self, values: List[float], decay_factor: float) -> float:
        """Calculate weighted average with exponential decay"""
        if not values:
            return 0.0
        weights = [decay_factor ** i for i in range(len(values) - 1, -1, -1)]
        return sum(w * v for w, v in zip(weights, values)) / sum(weights)

    def check_ngram_repetition(self, tokens: List[int]) -> bool:
        """Check for repeated n-grams in recent tokens"""
        window = tokens[-self.sliding_window:]
        for n in range(2, self.config.max_ngram_size + 1):
            ngrams = [tuple(window[i:i+n]) for i in range(len(window)-n+1)]
            for ngram in ngrams:
                if ngram in self.ngram_counts:
                    self.ngram_counts[ngram] += 1
                    if self.ngram_counts[ngram] > self.config.max_ngram_repeat:
                        return True
                else:
                    self.ngram_counts[ngram] = 1
        return False

    def determine_strategy(self, entropy: float, varentropy: float, attention_entropy: float) -> SamplerState:
        """Determine sampling strategy based on current metrics"""
        # Check for end of text condition first
        recent_tokens = list(self.recent_tokens)[-1:] if self.recent_tokens else []
        if recent_tokens and self.config.stop_tokens and any(token in self.config.stop_tokens for token in recent_tokens):
            return SamplerState.EOT

        # Standard strategy determination
        if entropy < self.config.low_ent_thresh and varentropy < self.config.low_vent_thresh:
            return SamplerState.ARGMAX
        elif entropy > self.config.med_ent_thresh and varentropy < self.config.varentropy_threshold:
            return SamplerState.INSERT_COT
        elif entropy < self.config.high_ent_thresh and varentropy > self.config.high_vent_thresh:
            return SamplerState.RESAMPLE
        elif entropy > self.config.med_ent_thresh and varentropy > self.config.high_vent_thresh:
            return SamplerState.ADAPTIVE
        else:
            return SamplerState.SAMPLE

    def sample(self, logits: torch.Tensor, attention: torch.Tensor) -> Tuple[torch.Tensor, SamplerState]:
        """Main sampling function with strategy selection and metric tracking"""
        metrics = self.calculate_metrics(logits, attention)
        
        # Update metric windows
        self.entropy_window.append(metrics["logits_entropy"])
        self.varentropy_window.append(metrics["logits_varentropy"])
        self.attention_entropy_window.append(metrics["attn_entropy"])
        self.long_entropy_window.append(metrics["logits_entropy"])
        self.long_varentropy_window.append(metrics["logits_varentropy"])
        
        # Calculate weighted averages
        avg_entropy = self.weighted_average(list(self.entropy_window), self.config.decay_factor)
        avg_varentropy = self.weighted_average(list(self.varentropy_window), self.config.decay_factor)
        avg_attention_entropy = self.weighted_average(list(self.attention_entropy_window), self.config.decay_factor)
        
        # Long-term metrics
        long_avg_entropy = self.weighted_average(list(self.long_entropy_window), self.config.long_decay_factor)
        long_avg_varentropy = self.weighted_average(list(self.long_varentropy_window), self.config.long_decay_factor)
        
        # Combine short and long-term metrics
        combined_entropy = (avg_entropy + long_avg_entropy) / 2
        combined_varentropy = (avg_varentropy + long_avg_varentropy) / 2
        
        # Determine sampling strategy
        self.current_strategy = self.determine_strategy(
            combined_entropy,
            combined_varentropy,
            avg_attention_entropy
        )
        
        # Sample based on current strategy
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
        
        # Update tracking
        self.strategy_counter[self.current_strategy.name] += 1
        self.tokens_since_last_change += 1
        self.current_batch.append(sampled_token.item())
        self.recent_tokens.append(sampled_token.item())
        
        # Handle repetition
        if self.check_ngram_repetition(list(self.recent_tokens)):
            sampled_token = self._sample(
                logits,
                temperature=1.2,
                top_k=100
            )
        
        # Reset batch if needed
        if len(self.current_batch) >= self.config.strategy_change_batch_size:
            self.current_batch = []
            self.tokens_since_last_change = 0
        
        return sampled_token, self.current_strategy

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 1000) -> str:
    """Generate response using the enhanced sampling strategy"""
    cfg = SamplerConfig(tokenizer)
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
    print("Generated text:", flush=True)
    
    with torch.inference_mode():
        for i in range(max_tokens):
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                output_attentions=True
            )
            
            logits = outputs.logits
            attention = outputs.attentions[-1]
            
            sampled_token, state = sampler.sample(logits, attention)
            
            if state == SamplerState.EOT or sampled_token[0] == tokenizer.eos_token_id:
                break
            
            # Add visible markers for COT insertions
            if state == SamplerState.INSERT_COT and sampled_token[0] == cfg.cot_token:
                next_token_text = "🤔[Let me think about this]"  # Visual marker for COT
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
    
    total_tokens = sum(sampler.strategy_counter.values())
    logger.info("\nToken Generation Strategy Distribution:")
    for strategy, count in sampler.strategy_counter.items():
        percentage = (count / total_tokens) * 100
        logger.info(f"{strategy}: {count} ({percentage:.2f}%)")
    
    return generated_text

def main():
    """Main function to run the enhanced text generation"""
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
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