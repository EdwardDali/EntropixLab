import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from enum import Enum
from typing import List, Tuple, Optional, Dict
import time
import numpy as np
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
        # Updated values to match sampler.py
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

        # Additional parameters from the original main_t.py
        self.cot_token = tokenizer.encode("[COT]", add_special_tokens=False)[0] if tokenizer else None
        self.repetition_penalty = 1.2
        self.max_ngram_size = 5
        self.max_ngram_repeat = 3
        self.strategy_change_batch_size = 1
        self.window_size = 50
        self.long_window_size = 500
        self.decay_factor = 0.95
        self.long_decay_factor = 0.95

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

    def sample(self, logits: torch.Tensor, attention: torch.Tensor) -> Tuple[torch.Tensor, SamplerState]:
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)
        attention_entropy = self.calculate_attention_entropy(attention)
        
        self.entropy_window.append(entropy.item())
        self.varentropy_window.append(varentropy.item())
        self.attention_entropy_window.append(attention_entropy.item())
        self.long_entropy_window.append(entropy.item())
        self.long_varentropy_window.append(varentropy.item())
        
        avg_entropy = self.weighted_average(self.entropy_window, self.config.decay_factor)
        avg_varentropy = self.weighted_average(self.varentropy_window, self.config.decay_factor)
        avg_attention_entropy = self.weighted_average(self.attention_entropy_window, self.config.decay_factor)
        long_avg_entropy = self.weighted_average(self.long_entropy_window, self.config.long_decay_factor)
        long_avg_varentropy = self.weighted_average(self.long_varentropy_window, self.config.long_decay_factor)
        
        combined_entropy = (avg_entropy + long_avg_entropy) / 2
        combined_varentropy = (avg_varentropy + long_avg_varentropy) / 2
        
        self.current_strategy = self.determine_strategy(combined_entropy, combined_varentropy, avg_attention_entropy)
        
        if self.current_strategy == SamplerState.ARGMAX:
            sampled_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        elif self.current_strategy == SamplerState.INSERT_COT:
            if self.config.cot_token not in self.recent_tokens:
                sampled_token = torch.tensor([[self.config.cot_token]], device=logits.device)
            else:
                temp_adj = self.config.helv_attn_ent_offset + self.config.helv_attn_ent_coef * avg_attention_entropy
                sampled_token = self._sample(logits, temperature=min(1.5, self.config.temp * temp_adj))
        elif self.current_strategy == SamplerState.RESAMPLE:
            temp_adj = self.config.lehv_interaction_strength_offset + self.config.lehv_interaction_strength_coef * self.calculate_interaction_strength(attention)
            top_k_adj = max(5, int(self.config.top_k * (1 + 0.5 * (1 - self.calculate_agreement(attention)))))
            sampled_token = self._sample(logits, temperature=min(1.5, self.config.temp * temp_adj), top_k=top_k_adj)
        elif self.current_strategy == SamplerState.ADAPTIVE:
            sampled_token = self._adaptive_sample(logits, attention)
        else:  # SamplerState.SAMPLE
            sampled_token = self._sample(logits)
        
        self.strategy_counter[self.current_strategy.name] += 1
        self.tokens_since_last_change += 1
        self.current_batch.append(sampled_token.item())
        self.recent_tokens.append(sampled_token.item())
        
        if self.check_ngram_repetition(list(self.recent_tokens)):
            sampled_token = self._sample(logits, temperature=1.2, top_k=100)
        
        if len(self.current_batch) == self.config.strategy_change_batch_size:
            self.current_batch = []
        
        return sampled_token, self.current_strategy

    def weighted_average(self, values, decay_factor):
        if not values:
            return 0
        weights = [decay_factor ** i for i in range(len(values) - 1, -1, -1)]
        return sum(w * v for w, v in zip(weights, values)) / sum(weights)

    def determine_strategy(self, entropy: float, varentropy: float, attention_entropy: float) -> SamplerState:
        if entropy < self.config.low_ent_thresh and varentropy < self.config.low_vent_thresh:
            return SamplerState.ARGMAX
        elif entropy > self.config.high_ent_thresh and varentropy < self.config.low_vent_thresh:
            return SamplerState.INSERT_COT
        elif entropy < self.config.high_ent_thresh and varentropy > self.config.high_vent_thresh:
            return SamplerState.RESAMPLE
        elif entropy > self.config.med_ent_thresh and varentropy > self.config.high_vent_thresh:
            return SamplerState.ADAPTIVE
        else:
            return SamplerState.SAMPLE

    def calculate_varentropy_logsoftmax(self, logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(logits, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2
        varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
        return entropy, varentropy

    def calculate_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        attention = attention.mean(dim=1)
        attention_probs = attention / attention.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(attention_probs * torch.log2(attention_probs + 1e-10), dim=-1)
        return entropy.mean()

    def calculate_interaction_strength(self, attention: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(attention))

    def calculate_agreement(self, attention: torch.Tensor) -> torch.Tensor:
        mean_attention = torch.mean(attention, dim=1)
        return 1 - torch.mean(torch.abs(attention - mean_attention.unsqueeze(1)))

    def _sample(self, logits: torch.Tensor, temperature: float = None, top_p: float = None, top_k: int = None, min_p: float = None) -> torch.Tensor:
        temperature = temperature or self.config.temp
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        min_p = min_p or self.config.min_p

        probs = F.softmax(logits / temperature, dim=-1)

        if min_p > 0.0:
            p_max = torch.max(probs)
            probs[probs < (min_p * p_max)] = 0
            probs = probs / probs.sum()

        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        
        cumulative_probs = torch.cumsum(top_k_probs, dim=-1)
        probs_to_keep = cumulative_probs <= top_p
        if not probs_to_keep.any():
            probs_to_keep[-1] = True
        top_k_probs = top_k_probs[probs_to_keep]
        top_k_indices = top_k_indices[probs_to_keep]

        if top_k_probs.sum() <= 0:
            return torch.argmax(probs).unsqueeze(0)

        try:
            sample = torch.multinomial(top_k_probs, num_samples=1)
            return top_k_indices[sample]
        except RuntimeError:
            return torch.argmax(probs).unsqueeze(0)

    def _adaptive_sample(self, logits: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        metrics = self.calculate_metrics(logits, attention)
        
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

    def calculate_metrics(self, logits: torch.Tensor, attention: torch.Tensor) -> Dict[str, float]:
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)
        attn_entropy = self.calculate_attention_entropy(attention)
        attn_varentropy = torch.var(attn_entropy)
        agreement = self.calculate_agreement(attention)
        interaction_strength = self.calculate_interaction_strength(attention)

        return {
            "logits_entropy": entropy.mean().item(),
            "logits_varentropy": varentropy.mean().item(),
            "attn_entropy": attn_entropy.item(),
            "attn_varentropy": attn_varentropy.item(),
            "agreement": agreement.item(),
            "interaction_strength": interaction_strength.item(),
            "logits_uncertainty": entropy.mean().item() + varentropy.mean().item(),
            "attn_uncertainty": attn_entropy.item() + attn_varentropy.item()
        }

    def score_sample(self, sample: torch.Tensor, logits: torch.Tensor, metrics: Dict[str, float]) -> float:
        log_prob = F.log_softmax(logits, dim=-1)[0, sample.item()].item()
        confidence_score = (
            (1 - metrics["logits_entropy"]) * self.config.ada_score_logits_ent +
            (1 - metrics["attn_entropy"]) * self.config.ada_score_attn_ent +
            (1 - metrics["logits_varentropy"]) * self.config.ada_score_logits_vent +
            (1 - metrics["attn_varentropy"]) * self.config.ada_score_attn_vent +
            metrics["agreement"] * self.config.ada_score_agree +
            metrics["interaction_strength"] * self.config.ada_score_int
        )
        return log_prob + confidence_score

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
            
            input_ids = torch.cat([input_ids, sampled_token.unsqueeze(0)], dim=-1)
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