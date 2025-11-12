import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class TokenProbabilityExplorer:

    def __init__(self, model_name: str = "gpt2"):
   
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def get_next_token_probabilities(
        self, 
        prompt: str, 
        top_k: int = 10,
        temperature: float = 1.0
    ) -> Tuple[List[str], List[float], List[int]]:

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]  # Get logits for last token
        
        # Apply temperature
        logits = logits / temperature
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        
        # Decode tokens
        tokens = [self.tokenizer.decode([idx]) for idx in top_indices]
        probabilities = top_probs.cpu().numpy()
        token_ids = top_indices.cpu().numpy()
        
        return tokens, probabilities, token_ids
    
    def visualize_probabilities(
        self, 
        prompt: str, 
        top_k: int = 15,
        temperature: float = 1.0,
        save_path: str = None
    ):

        tokens, probs, token_ids = self.get_next_token_probabilities(
            prompt, top_k, temperature
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar chart
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(tokens)))
        bars = ax.barh(range(len(tokens)), probs, color=colors)
        
        # Customize plot
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels([f"'{t}'" for t in tokens])
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title(
            f'Top {top_k} Next Token Predictions\n'
            f'Prompt: "{prompt}"\n'
            f'Temperature: {temperature}',
            fontsize=14,
            pad=20
        )
        ax.grid(axis='x', alpha=0.3)
        
        # Add probability labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            width = bar.get_width()
            ax.text(
                width, bar.get_y() + bar.get_height()/2,
                f' {prob:.4f}',
                ha='left', va='center',
                fontsize=10
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def compare_temperatures(
        self, 
        prompt: str, 
        temperatures: List[float] = [0.5, 1.0, 2.0],
        top_k: int = 10
    ):

        fig, axes = plt.subplots(1, len(temperatures), figsize=(6*len(temperatures), 8))
        
        if len(temperatures) == 1:
            axes = [axes]
        
        for ax, temp in zip(axes, temperatures):
            tokens, probs, _ = self.get_next_token_probabilities(
                prompt, top_k, temp
            )
            
            colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(tokens)))
            bars = ax.barh(range(len(tokens)), probs, color=colors)
            
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels([f"'{t}'" for t in tokens])
            ax.set_xlabel('Probability', fontsize=11)
            ax.set_title(f'Temperature: {temp}', fontsize=12, pad=10)
            ax.grid(axis='x', alpha=0.3)
            
            # Add probability labels
            for bar, prob in zip(bars, probs):
                width = bar.get_width()
                ax.text(
                    width, bar.get_y() + bar.get_height()/2,
                    f' {prob:.3f}',
                    ha='left', va='center',
                    fontsize=9
                )
        
        fig.suptitle(
            f'Temperature Comparison for: "{prompt}"',
            fontsize=14,
            y=1.02
        )
        plt.tight_layout()
        plt.show()
    
    def generate_with_analysis(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k_display: int = 5
    ) -> Dict:

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        
        generation_history = []
        current_text = prompt
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
            
            # Get top-k for display
            top_probs, top_indices = torch.topk(probs, top_k_display)
            
            # Sample next token
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Record step information
            step_info = {
                'step': step + 1,
                'selected_token': self.tokenizer.decode([next_token_id]),
                'selected_prob': probs[next_token_id].item(),
                'alternatives': [
                    {
                        'token': self.tokenizer.decode([idx]),
                        'prob': prob.item()
                    }
                    for idx, prob in zip(top_indices, top_probs)
                ]
            }
            generation_history.append(step_info)
            
            # Update input_ids
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            current_text += step_info['selected_token']
            
            # Stop if EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                break
        
        return {
            'prompt': prompt,
            'generated_text': current_text,
            'history': generation_history,
            'total_tokens': len(generation_history)
        }
    
    def print_generation_analysis(self, analysis: Dict):
   
        print("=" * 80)
        print("GENERATION ANALYSIS")
        print("=" * 80)
        print(f"\nPrompt: {analysis['prompt']}")
        print(f"Generated: {analysis['generated_text']}")
        print(f"\nTotal tokens generated: {analysis['total_tokens']}")
        print("\n" + "-" * 80)
        print("Step-by-step breakdown:")
        print("-" * 80)
        
        for step_info in analysis['history']:
            print(f"\nStep {step_info['step']}:")
            print(f"  Selected: '{step_info['selected_token']}' "
                  f"(prob: {step_info['selected_prob']:.4f})")
            print(f"  Top alternatives:")
            for i, alt in enumerate(step_info['alternatives'][:3], 1):
                print(f"    {i}. '{alt['token']}' (prob: {alt['prob']:.4f})")


def main():

    print("🚀 Token Probability Explorer")
    print("=" * 80)
    
    # Initialize explorer with a smaller model (faster loading)
    explorer = TokenProbabilityExplorer(model_name="gpt2")
    
    # Example 1: Visualize probabilities for a simple prompt
    print("\n📊 Example 1: Visualizing next token probabilities")
    prompt1 = "The capital of France is"
    explorer.visualize_probabilities(prompt1, top_k=15, temperature=1.0)
    
    # Example 2: Compare different temperatures
    print("\n🌡️  Example 2: Temperature comparison")
    prompt2 = "Once upon a time"
    explorer.compare_temperatures(prompt2, temperatures=[0.5, 1.0, 2.0], top_k=10)
    
    # Example 3: Generate with analysis
    print("\n🔍 Example 3: Generation with step-by-step analysis")
    prompt3 = "The future of AI is"
    analysis = explorer.generate_with_analysis(
        prompt3,
        max_new_tokens=10,
        temperature=0.8,
        top_k_display=5
    )
    explorer.print_generation_analysis(analysis)
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
