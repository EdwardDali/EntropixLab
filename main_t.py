import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from enum import Enum
from typing import List, Tuple, Optional, Dict
import time
from collections import Counter, deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

class SamplerState(Enum):
    ARGMAX = 0
    SAMPLE = 1
    INSERT_COT = 2
    RESAMPLE = 3
    ADAPTIVE = 4

class SamplerConfig:
    def __init__(self, tokenizer=None):
        self.temp = 0.666
        self.top_p = 0.90
        self.top_k = 27
        self.min_p = 0.03

        self.low_ent_thresh = 0.1
        self.low_vent_thresh = 0.1
        self.med_ent_thresh = 3.0
        self.high_ent_thresh = 5.0
        self.high_vent_thresh = 5.0

        self.helv_attn_ent_offset = 1.3
        self.helv_attn_ent_coef = 0.2

        self.lehv_interaction_strength_offset = 1.2
        self.lehv_interaction_strength_coef = 0.3

        self.hehv_attn_ent_coef = 0.2
        self.hehv_attn_vent_offset = 2.0
        self.hehv_attn_vent_coef = 0.5

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

        self.cot_token = tokenizer.encode("[COT]", add_special_tokens=False)[0] if tokenizer else None
        self.clarifying_question_token = tokenizer.encode("?", add_special_tokens=False)[0] if tokenizer else None
        self.repetition_penalty = 1.2
        self.max_ngram_size = 5
        self.max_ngram_repeat = 3
        self.strategy_change_batch_size = 1
        self.window_size = 50
        self.long_window_size = 500
        self.decay_factor = 0.95
        self.long_decay_factor = 0.95

        self.generator = torch.Generator(device=device).manual_seed(1337)

        # Align with the approach from torch_sampler.py where 2564 is used
        self.cot_token = 2564  # Default COT token ID
        self.question_threshold = 3.0  # From torch_sampler's entropy threshold
        self.varentropy_threshold = 0.1  # From torch_sampler's varentropy threshold
        
        # Keep existing config parameters but rename for clarity
        self.helv_temp_base = 1.3  # Base temperature adjustment for high entropy
        self.helv_attn_coef = 0.2  # Attention entropy coefficient
        
        if tokenizer:
            # Try to get token ID from tokenizer if available, otherwise use default
            try:
                self.cot_token = tokenizer.encode("[COT]", add_special_tokens=False)[0]
            except:
                logger.warning("Could not encode [COT] token, using default ID 2564")

class EntropixSampler:
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.strategy_counter = Counter()
        self.recent_tokens = deque(maxlen=200)
        self.current_batch = []
        self.current_strategy = SamplerState.SAMPLE
        self.tokens_since_last_change = 0
        self.entropy_window = deque(maxlen=self.config.window_size)
        self.varentropy_window = deque(maxlen=self.config.window_size)
        self.attention_entropy_window = deque(maxlen=self.config.window_size)
        self.long_entropy_window = deque(maxlen=self.config.long_window_size)
        self.long_varentropy_window = deque(maxlen=self.config.long_window_size)
        self.ngram_counts = {}
        self.sliding_window = 100

    def calculate_varentropy_logsoftmax(self, logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(logits, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2
        varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
        return entropy, varentropy

    def calculate_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        attention_probs = F.softmax(attention, dim=-1)
        entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, min=1e-10)), dim=-1)
        return entropy.mean(dim=1)

    def calculate_attention_varentropy(self, attention: torch.Tensor) -> torch.Tensor:
        attention_entropy = self.calculate_attention_entropy(attention)
        varentropy = torch.var(attention_entropy)
        return torch.where(torch.isnan(varentropy), torch.zeros_like(varentropy), varentropy)

    def calculate_agreement(self, attention: torch.Tensor) -> torch.Tensor:
        attention_probs = F.softmax(attention, dim=-1)
        mean_attention = torch.mean(attention_probs, dim=1, keepdim=True)
        return 1 - torch.mean(torch.abs(attention_probs - mean_attention))

    def calculate_interaction_strength(self, attention: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(attention))

    def calculate_metrics(self, logits: torch.Tensor, attention: torch.Tensor) -> Dict[str, float]:
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

    def _sample(self, logits: torch.Tensor, temperature: float = None, top_p: float = None, top_k: int = None, min_p: float = None) -> torch.Tensor:
        temperature = temperature or self.config.temp
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        min_p = min_p or self.config.min_p

        if logits.dim() == 3:
            logits = logits[:, -1, :]
        elif logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        logits = logits / temperature

        if min_p > 0.0:
            sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > (1 - min_p)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=torch.argsort(logits, descending=True), src=sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        top_k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
        
        cumulative_probs = torch.cumsum(F.softmax(top_k_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        top_k_logits = top_k_logits.masked_fill(sorted_indices_to_remove, float('-inf'))

        probs = F.softmax(top_k_logits, dim=-1)
        sample = torch.multinomial(probs, num_samples=1, generator=self.config.generator)
        return top_k_indices.gather(-1, sample)

    def _adaptive_sample(self, logits: torch.Tensor, metrics: Dict[str, float]) -> torch.Tensor:
        temperature = self.config.temp * (1 + self.config.ada_temp_logits * metrics["logits_uncertainty"] + 
                                          self.config.ada_temp_attn * metrics["attn_uncertainty"] - 
                                          self.config.ada_temp_agree * metrics["agreement"])
        top_p = min(max(self.config.top_p * (1 + self.config.ada_top_p * metrics["attn_varentropy"]), 0.1), 1.0)
        top_k = int(min(max(self.config.top_k * (1 + self.config.ada_top_k_int * metrics["interaction_strength"] - 
                                                 self.config.ada_top_k_agree * metrics["agreement"]), 1), 100))
        min_p = min(max(self.config.min_p * (1 - self.config.ada_min_p * metrics["logits_uncertainty"]), 0.01), 0.5)

        samples = []
        for _ in range(self.config.n_adaptive_samples):
            sample = self._sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
            samples.append(sample)

        sample_scores = [self.score_sample(sample, logits, metrics) for sample in samples]
        best_sample_idx = torch.argmax(torch.tensor(sample_scores))
        return samples[best_sample_idx]

    def score_sample(self, sample: torch.Tensor, logits: torch.Tensor, metrics: Dict[str, float]) -> float:
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

    def weighted_average(self, values, decay_factor):
        if not values:
            return 0
        weights = [decay_factor ** i for i in range(len(values) - 1, -1, -1)]
        return sum(w * v for w, v in zip(weights, values)) / sum(weights)

    def determine_strategy(self, entropy: float, varentropy: float, attention_entropy: float) -> SamplerState:
        # Align with torch_sampler.py conditions
        if entropy < self.config.low_ent_thresh and varentropy < self.config.low_vent_thresh:
            return SamplerState.ARGMAX
        elif entropy > self.config.question_threshold and varentropy < self.config.varentropy_threshold:
            # Align with torch_sampler.py's "High Entropy, Low Varentropy" condition
            return SamplerState.INSERT_COT
        elif entropy < self.config.high_ent_thresh and varentropy > self.config.high_vent_thresh:
            return SamplerState.RESAMPLE
        elif entropy > self.config.med_ent_thresh and varentropy > self.config.high_vent_thresh:
            return SamplerState.ADAPTIVE
        else:
            return SamplerState.SAMPLE

    def check_ngram_repetition(self, tokens: List[int]) -> bool:
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

    def sample(self, logits: torch.Tensor, attention: torch.Tensor) -> Tuple[torch.Tensor, SamplerState]:
        """
        Sample next token using entropy-guided state machine with aligned behaviors.
        
        Args:
            logits: Token logits of shape (batch_size, vocab_size) or (batch_size, sequence_length, vocab_size)
            attention: Attention scores from the last layer
            
        Returns:
            Tuple of (sampled token tensor, current sampling strategy)
        """
        metrics = self.calculate_metrics(logits, attention)
        
        # Update rolling windows for entropy metrics
        self.entropy_window.append(metrics["logits_entropy"])
        self.varentropy_window.append(metrics["logits_varentropy"])
        self.attention_entropy_window.append(metrics["attn_entropy"])
        self.long_entropy_window.append(metrics["logits_entropy"])
        self.long_varentropy_window.append(metrics["logits_varentropy"])
        
        # Calculate weighted averages for short and long-term metrics
        avg_entropy = self.weighted_average(self.entropy_window, self.config.decay_factor)
        avg_varentropy = self.weighted_average(self.varentropy_window, self.config.decay_factor)
        avg_attention_entropy = self.weighted_average(self.attention_entropy_window, self.config.decay_factor)
        long_avg_entropy = self.weighted_average(self.long_entropy_window, self.config.long_decay_factor)
        long_avg_varentropy = self.weighted_average(self.long_varentropy_window, self.config.long_decay_factor)
        
        # Combine short and long-term metrics
        combined_entropy = (avg_entropy + long_avg_entropy) / 2
        combined_varentropy = (avg_varentropy + long_avg_varentropy) / 2
        
        # Determine sampling strategy based on metrics
        self.current_strategy = self.determine_strategy(combined_entropy, combined_varentropy, avg_attention_entropy)
        
        # Sample based on current strategy
        if self.current_strategy == SamplerState.ARGMAX:
            # Low entropy, low varentropy: confident prediction
            sampled_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        
        elif self.current_strategy == SamplerState.INSERT_COT:
            # High entropy, low varentropy: needs clarification
            # Align with torch_sampler's COT insertion logic
            if self.config.cot_token not in self.recent_tokens:
                sampled_token = torch.tensor([[self.config.cot_token]], device=logits.device)
            else:
                # If we just used COT, sample with adjusted temperature
                temp_adj = self.config.helv_attn_ent_offset + self.config.helv_attn_ent_coef * avg_attention_entropy
                sampled_token = self._sample(
                    logits, 
                    temperature=min(1.5, self.config.temp * temp_adj)
                )
        
        elif self.current_strategy == SamplerState.RESAMPLE:
            # Low entropy, high varentropy: exploring alternatives
            temp_adj = (self.config.lehv_interaction_strength_offset + 
                    self.config.lehv_interaction_strength_coef * metrics["interaction_strength"])
            top_k_adj = max(5, int(self.config.top_k * (1 + 0.5 * (1 - metrics["agreement"]))))
            sampled_token = self._sample(
                logits,
                temperature=min(1.5, self.config.temp * temp_adj),
                top_k=top_k_adj
            )
        
        elif self.current_strategy == SamplerState.ADAPTIVE:
            # High entropy, high varentropy: need careful sampling
            sampled_token = self._adaptive_sample(logits, metrics)
        
        else:  # SamplerState.SAMPLE
            # Default sampling behavior
            sampled_token = self._sample(logits)
        
        # Update tracking counters and history
        self.strategy_counter[self.current_strategy.name] += 1
        self.tokens_since_last_change += 1
        self.current_batch.append(sampled_token.item())
        self.recent_tokens.append(sampled_token.item())
        
        # Check for and handle repetition
        if self.check_ngram_repetition(list(self.recent_tokens)):
            # If repetitive, resample with higher temperature and larger top-k
            sampled_token = self._sample(
                logits,
                temperature=1.2,
                top_k=100
            )
        
        # Reset batch if needed
        if len(self.current_batch) == self.config.strategy_change_batch_size:
            self.current_batch = []
        
        # Apply repetition penalty if configured
        if self.config.repetition_penalty != 1.0 and len(self.recent_tokens) > 0:
            logits[:, self.recent_tokens] = logits[:, self.recent_tokens] / self.config.repetition_penalty
        
        return sampled_token, self.current_strategy

def generate_response(model, tokenizer, prompt, max_tokens=1000):
    cfg = SamplerConfig(tokenizer)
    sampler = EntropixSampler(cfg)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings - max_tokens).to(device)
    attention_mask = torch.ones_like(input_ids)
    
    generated_text = ""
    start_time = time.time()
    
    print(f"Generating response for prompt: '{prompt}'")
    print("Generated text:")
    
    for i in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
            logits = outputs.logits[:, -1, :]
            attention = outputs.attentions[-1]
            
            sampled_token, state = sampler.sample(logits, attention)
            
            if sampled_token[0] == tokenizer.eos_token_id:
                print("\nEnd of sequence token generated. Stopping.")
                break
            
            next_token_text = tokenizer.decode(sampled_token[0])
            generated_text += next_token_text
            print(next_token_text, end="", flush=True)
            
            if "<|endoftext|>" in generated_text:
                print("\n<|endoftext|> token encountered. Stopping generation.")
                generated_text = generated_text.split("<|endoftext|>")[0]
                break
            
            input_ids = torch.cat([input_ids, sampled_token.transpose(0, 1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
        
        if input_ids.shape[1] >= model.config.max_position_embeddings:
            print("\nReached maximum sequence length. Stopping.")
            break
    
    total_time = time.time() - start_time
    print(f"\n\nGeneration completed in {total_time:.2f} seconds.")
    
    total_tokens = sum(sampler.strategy_counter.values())
    print("\nToken Generation Strategy Distribution:")
    for strategy, count in sampler.strategy_counter.items():
        percentage = (count / total_tokens) * 100
        print(f"{strategy}: {count} ({percentage:.2f}%)")
    
    return generated_text

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(device)
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
        
        logger.info(f"Generating response for prompt: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        
        print(f"\nPrompt: {prompt}")
        print(f"Generated response: {response}")
        print("\n" + "-"*50 + "\n")
        
        with open("generated_response.txt", "w", encoding="utf-8") as file:
            file.write(f"Prompt: {prompt}\n\nGenerated response: {response}")
        print("Response saved to generated_response.txt")
    
    print("Thank you for using the AI assistant. Goodbye!")

if __name__ == "__main__":
    main()