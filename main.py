import torch
from torch.nn import functional as F
from llama_cpp import Llama, LogitsProcessorList, LogitsProcessor
from typing import List, Tuple, Optional, Dict
import logging
import os
from enum import Enum
import time
import numpy as np
from collections import Counter, deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cpu")  # llama-cpp-python uses CPU by default

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

class SamplerState(Enum):
    ARGMAX = 0
    SAMPLE = 1
    INSERT_COT = 2
    RESAMPLE = 3
    ADAPTIVE = 4  # New adaptive sampling strategy

class SamplerConfig:
    def __init__(self):
        self.entropy_threshold = 1.0
        self.varentropy_threshold = 1.5
        self.cot_token = "[COT]"
        self.resample_count = 5
        self.strategy_params: Dict[SamplerState, Dict[str, float]] = {
            SamplerState.ARGMAX: {"temperature": 0.1, "top_p": 1.0, "top_k": 1, "min_p": 0.0},
            SamplerState.SAMPLE: {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "min_p": 0.02},
            SamplerState.INSERT_COT: {"temperature": 0.8, "top_p": 0.95, "top_k": 100, "min_p": 0.01},
            SamplerState.RESAMPLE: {"temperature": 1.0, "top_p": 0.98, "top_k": 200, "min_p": 0.005},
            SamplerState.ADAPTIVE: {"temperature": 0.666, "top_p": 0.90, "top_k": 27, "min_p": 0.03}
        }
        self.repetition_penalty = 1.2
        self.max_ngram_size = 5
        self.max_ngram_repeat = 3
        self.strategy_change_batch_size = 5
        self.window_size = 50  # Size of the sliding window for weighted average
        self.decay_factor = 0.95  # Exponential decay factor for weighting
        
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

class VarentropyLogitsProcessor(LogitsProcessor):
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.strategy_counter = Counter()
        self.recent_tokens = deque(maxlen=100)
        self.current_batch = []
        self.current_strategy = SamplerState.SAMPLE
        self.tokens_since_last_change = 0
        self.entropy_window = deque(maxlen=self.config.window_size)
        self.varentropy_window = deque(maxlen=self.config.window_size)

    def __call__(self, input_ids: List[int], scores: List[float]) -> List[float]:
        logits = np.array(scores)
        
        # Calculate entropy and varentropy for the current token
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)
        self.entropy_window.append(entropy)
        self.varentropy_window.append(varentropy)
        
        # Check if it's time to recalculate the strategy
        if self.tokens_since_last_change % self.config.strategy_change_batch_size == 0:
            avg_entropy = self.weighted_average(self.entropy_window)
            avg_varentropy = self.weighted_average(self.varentropy_window)
            
            self.current_strategy = self.determine_strategy(avg_entropy, avg_varentropy)
            self.tokens_since_last_change = 0
        
        # Use the current strategy to sample
        if self.current_strategy == SamplerState.ADAPTIVE:
            sampled_token = self._adaptive_sample(logits)
        else:
            params = self.config.strategy_params[self.current_strategy]
            sampled_token = self._sample(logits, **params)
        
        # Update counters and lists
        self.strategy_counter[self.current_strategy.name] += 1
        self.tokens_since_last_change += 1
        self.current_batch.append(sampled_token)
        self.recent_tokens.append(sampled_token)
        
        # Check for n-gram repetition in the current batch
        if self.check_ngram_repetition(self.current_batch):
            # Increase temperature and top_k to encourage diversity
            temp_config = SamplerConfig()
            temp_config.strategy_params[SamplerState.SAMPLE]["temperature"] = 1.2
            temp_config.strategy_params[SamplerState.SAMPLE]["top_k"] = 100
            sampled_token = self._sample(logits, **temp_config.strategy_params[SamplerState.SAMPLE])
        
        # Reset batch if it reaches the configured batch size
        if len(self.current_batch) == self.config.strategy_change_batch_size:
            self.current_batch = []
        
        # Set all logits to negative infinity except the sampled token
        new_scores = [-float('inf')] * len(scores)
        new_scores[sampled_token] = 0
        
        return new_scores

    def weighted_average(self, values):
        if not values:
            return 0
        weights = [self.config.decay_factor ** i for i in range(len(values) - 1, -1, -1)]
        return sum(w * v for w, v in zip(weights, values)) / sum(weights)

    def determine_strategy(self, entropy: float, varentropy: float) -> SamplerState:
        if entropy < self.config.entropy_threshold:
            if varentropy < self.config.varentropy_threshold:
                return SamplerState.ARGMAX
            else:
                return SamplerState.SAMPLE
        else:
            if varentropy < self.config.varentropy_threshold:
                return SamplerState.INSERT_COT
            elif varentropy > self.config.varentropy_threshold * 1.5:  # Adjust this threshold as needed
                return SamplerState.RESAMPLE
            else:
                return SamplerState.ADAPTIVE

    def calculate_varentropy_logsoftmax(self, logits: np.ndarray, axis: int = -1) -> tuple[float, float]:
        logits_tensor = torch.from_numpy(logits)
        log_probs = F.log_softmax(logits_tensor, dim=axis)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=axis) / np.log(2)  # Convert to base-2
        varentropy = torch.sum(probs * (log_probs / np.log(2) + entropy.unsqueeze(-1))**2, dim=axis)
        return entropy.item(), varentropy.item()

    def _sample(self, logits: np.ndarray, temperature: float, top_p: float, top_k: int, min_p: float) -> int:
        logits_tensor = torch.from_numpy(logits)
        probs = F.softmax(logits_tensor / temperature, dim=-1)

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
            return torch.argmax(probs).item()

        # Sample from the filtered distribution
        try:
            sample = torch.multinomial(top_k_probs, num_samples=1)
            return top_k_indices[sample].item()
        except RuntimeError:
            # If multinomial fails, fall back to argmax
            return torch.argmax(probs).item()

    def _adaptive_sample(self, logits: np.ndarray) -> int:
        # Calculate metrics (simplified version as we don't have access to attention scores)
        entropy, varentropy = self.calculate_varentropy_logsoftmax(logits)
        
        # Adaptive sampling parameters (using fixed values from config)
        temperature = self.config.strategy_params[SamplerState.ADAPTIVE]["temperature"]
        top_p = self.config.strategy_params[SamplerState.ADAPTIVE]["top_p"]
        top_k = self.config.strategy_params[SamplerState.ADAPTIVE]["top_k"]
        min_p = self.config.strategy_params[SamplerState.ADAPTIVE]["min_p"]

        # Sample multiple times
        samples = []
        for _ in range(self.config.n_adaptive_samples):
            sample = self._sample(logits, temperature, top_p, top_k, min_p)
            samples.append(sample)

        # Score samples (simplified version)
        def score_sample(sample):
            log_prob = np.log(F.softmax(torch.from_numpy(logits), dim=-1)[sample].item())
            confidence_score = (
                (1 - entropy) * self.config.ada_score_logits_ent +
                (1 - varentropy) * self.config.ada_score_logits_vent
            )
            return log_prob + confidence_score

        sample_scores = [score_sample(sample) for sample in samples]
        best_sample_idx = np.argmax(sample_scores)
        return samples[best_sample_idx]

    def check_ngram_repetition(self, tokens: List[int]) -> bool:
        for n in range(2, self.config.max_ngram_size + 1):
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            for ngram in set(ngrams):
                if ngrams.count(ngram) > self.config.max_ngram_repeat:
                    return True
        return False


def generate_response(model, prompt, max_tokens=None, batch_size=10):
    cfg = SamplerConfig()
    cfg.strategy_change_batch_size = batch_size  # Set the batch size
    logits_processor = VarentropyLogitsProcessor(cfg)
    logits_processors = LogitsProcessorList([logits_processor])
    
    start_time = time.time()
    
    print(f"Generating response for prompt: '{prompt}'")
    print("Generating tokens...")
    
    # Use default parameters for initial generation
    default_params = cfg.strategy_params[SamplerState.SAMPLE]
    
    generation_params = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "logits_processor": logits_processors,
        "echo": False,
        "temperature": default_params['temperature'],
        "top_p": default_params['top_p'],
        "top_k": default_params['top_k'],
        "stream": True,
    }
    
    generated_text = ""
    try:
        for output in model(**generation_params):
            token = output['choices'][0]['text']
            generated_text += token
            print(token, end='', flush=True)
            
            if '[STOP]' in generated_text:
                break
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
    
    total_time = time.time() - start_time
    print(f"\n\nGeneration completed in {total_time:.2f} seconds.")
    
    # Print the strategy distribution
    total_tokens = sum(logits_processor.strategy_counter.values())
    print("\nToken Generation Strategy Distribution:")
    for strategy, count in logits_processor.strategy_counter.items():
        percentage = (count / total_tokens) * 100
        print(f"{strategy}: {count} ({percentage:.2f}%)")
    
    return generated_text

def save_to_file(text, filename="generated_response.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Response saved to {filename}")

def get_user_input():
    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ").strip()
        if prompt.lower() == 'quit':
            return None
        if prompt:
            return prompt
        print("Please enter a non-empty prompt.")

def main():
    model_path = r"C:\Users\User\Desktop\entropix\qwen2.5-0.5b-instruct-q5_k_m.gguf"  # Replace with your GGUF model path
    logger.info(f"Loading model: {model_path}")
    
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=8192,  # Increase context size to allow for longer generations
            n_gpu_layers=-1,
            verbose=False
        )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    print("Type 'quit' to exit the program.")
    
    while True:
        prompt = get_user_input()
        if prompt is None:
            break
        
        logger.info(f"Generating response for prompt: {prompt}")
        response = generate_response(model, prompt, max_tokens=None)  # Set max_tokens to None for unlimited generation
        
        print(f"\nPrompt: {prompt}")
        print(f"Generated response: {response}")
        print("\n" + "-"*50 + "\n")
        
        # Save the response to a file
        save_to_file(f"Prompt: {prompt}\n\nGenerated response: {response}")
    
    print("Thank you for using the AI assistant. Goodbye!")

if __name__ == "__main__":
    main()