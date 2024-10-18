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

class SamplerConfig:
    def __init__(self, tokenizer):
        self.entropy_threshold = 1.0
        self.varentropy_threshold = 1.5
        self.attention_entropy_threshold = 2.0  # New threshold for attention entropy
        self.cot_token = tokenizer.encode("[COT]", add_special_tokens=False)[0]
        self.resample_count = 5
        self.strategy_params: Dict[SamplerState, Dict[str, float]] = {
            SamplerState.ARGMAX: {"temperature": 0.1, "top_p": 1.0, "top_k": 1, "min_p": 0.0},
            SamplerState.SAMPLE: {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "min_p": 0.02},
            SamplerState.INSERT_COT: {"temperature": 0.8, "top_p": 0.95, "top_k": 100, "min_p": 0.01},
            SamplerState.RESAMPLE: {"temperature": 1.0, "top_p": 0.98, "top_k": 200, "min_p": 0.005}
        }
        self.repetition_penalty = 1.2
        self.max_ngram_size = 5
        self.max_ngram_repeat = 3
        self.strategy_change_batch_size = 1
        self.window_size = 5  # Size of the sliding window for weighted average
        self.decay_factor = 0.95  # Exponential decay factor for weighting

class EntropixSampler:
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.strategy_counter = Counter()
        self.recent_tokens = deque(maxlen=100)
        self.current_batch = []
        self.current_strategy = SamplerState.SAMPLE
        self.tokens_since_last_change = 0
        self.entropy_window = deque(maxlen=self.config.window_size)
        self.varentropy_window = deque(maxlen=self.config.window_size)
        self.attention_entropy_window = deque(maxlen=self.config.window_size)

    def sample(self, logits: torch.Tensor, attention: torch.Tensor) -> Tuple[torch.Tensor, SamplerState]:
        # Calculate entropy, varentropy, and attention entropy for the current token
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)
        attention_entropy = self.calculate_attention_entropy(attention)
        
        self.entropy_window.append(entropy.item())
        self.varentropy_window.append(varentropy.item())
        self.attention_entropy_window.append(attention_entropy.item())
        
        # Check if it's time to recalculate the strategy
        if self.tokens_since_last_change % self.config.strategy_change_batch_size == 0:
            avg_entropy = self.weighted_average(self.entropy_window)
            avg_varentropy = self.weighted_average(self.varentropy_window)
            avg_attention_entropy = self.weighted_average(self.attention_entropy_window)
            
            self.current_strategy = self.determine_strategy(avg_entropy, avg_varentropy, avg_attention_entropy)
            self.tokens_since_last_change = 0
        
        # Use the current strategy to sample
        params = self.config.strategy_params[self.current_strategy]
        sampled_token = self._sample(logits, **params)
        
        # Update counters and lists
        self.strategy_counter[self.current_strategy.name] += 1
        self.tokens_since_last_change += 1
        self.current_batch.append(sampled_token.item())
        self.recent_tokens.append(sampled_token.item())
        
        # Check for n-gram repetition in the current batch
        if self.check_ngram_repetition(self.current_batch):
            # Increase temperature and top_k to encourage diversity
            temp_config = SamplerConfig(None)
            temp_config.strategy_params[SamplerState.SAMPLE]["temperature"] = 1.2
            temp_config.strategy_params[SamplerState.SAMPLE]["top_k"] = 100
            sampled_token = self._sample(logits, **temp_config.strategy_params[SamplerState.SAMPLE])
        
        # Reset batch if it reaches the configured batch size
        if len(self.current_batch) == self.config.strategy_change_batch_size:
            self.current_batch = []
        
        return sampled_token, self.current_strategy

    def weighted_average(self, values):
        if not values:
            return 0
        weights = [self.config.decay_factor ** i for i in range(len(values) - 1, -1, -1)]
        return sum(w * v for w, v in zip(weights, values)) / sum(weights)

    def determine_strategy(self, entropy: float, varentropy: float, attention_entropy: float) -> SamplerState:
        if entropy < self.config.entropy_threshold and attention_entropy < self.config.attention_entropy_threshold:
            return SamplerState.ARGMAX if varentropy < self.config.varentropy_threshold else SamplerState.SAMPLE
        elif entropy >= self.config.entropy_threshold and attention_entropy < self.config.attention_entropy_threshold:
            return SamplerState.INSERT_COT
        else:
            return SamplerState.RESAMPLE

    def calculate_varentropy_logsoftmax(self, logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(logits, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
        varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
        return entropy, varentropy

    def calculate_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        # Assuming attention is of shape (batch_size, num_heads, seq_len, seq_len)
        # We'll calculate entropy for each head and then average
        attention = attention.mean(dim=1)  # Average over heads
        attention_probs = attention / attention.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(attention_probs * torch.log2(attention_probs + 1e-10), dim=-1)
        return entropy.mean()  # Average over sequence length

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int, min_p: float) -> torch.Tensor:
        probs = F.softmax(logits / temperature, dim=-1)

        # Apply min_p sampling
        if min_p > 0.0:
            p_max = torch.max(probs)
            probs[probs < (min_p * p_max)] = 0
            probs = probs / probs.sum()

        # Apply top-k sampling
        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        
        # Apply top-p sampling
        cumulative_probs = torch.cumsum(top_k_probs, dim=-1)
        probs_to_keep = cumulative_probs <= top_p
        if not probs_to_keep.any():
            probs_to_keep[-1] = True
        top_k_probs = top_k_probs[probs_to_keep]
        top_k_indices = top_k_indices[probs_to_keep]

        # Ensure we have valid probabilities
        if top_k_probs.sum() <= 0:
            return torch.argmax(probs).unsqueeze(0)

        # Sample from the filtered distribution
        try:
            sample = torch.multinomial(top_k_probs, num_samples=1)
            return top_k_indices[sample]
        except RuntimeError:
            # If multinomial fails, fall back to argmax
            return torch.argmax(probs).unsqueeze(0)

    def check_ngram_repetition(self, tokens: List[int]) -> bool:
        for n in range(2, self.config.max_ngram_size + 1):
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            for ngram in set(ngrams):
                if ngrams.count(ngram) > self.config.max_ngram_repeat:
                    return True
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
            attention = outputs.attentions[-1]  # Get the last layer's attention
            
            sampled_token, state = sampler.sample(logits, attention)
            
            if sampled_token[0] == tokenizer.eos_token_id:
                print("\nEnd of sequence token generated. Stopping.")
                break
            
            next_token_text = tokenizer.decode(sampled_token[0])
            generated_text += next_token_text
            print(next_token_text, end="", flush=True)
            
            # Check for "<|endoftext|>" token
            if "<|endoftext|>" in generated_text:
                print("\n<|endoftext|> token encountered. Stopping generation.")
                generated_text = generated_text.split("<|endoftext|>")[0]  # Keep only the text before the token
                break
            
            input_ids = torch.cat([input_ids, sampled_token.unsqueeze(0)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
        
        if input_ids.shape[1] >= model.config.max_position_embeddings:
            print("\nReached maximum sequence length. Stopping.")
            break
    
    total_time = time.time() - start_time
    print(f"\n\nGeneration completed in {total_time:.2f} seconds.")
    
    # Print the strategy distribution
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
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
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
        
        # Save the response to a file
        with open("generated_response.txt", "w", encoding="utf-8") as file:
            file.write(f"Prompt: {prompt}\n\nGenerated response: {response}")
        print("Response saved to generated_response.txt")
    
    print("Thank you for using the AI assistant. Goodbye!")

if __name__ == "__main__":
    main()