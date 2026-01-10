"""
Debug script to analyze number of steps ("\n\n" separators) in dataset.
Usage: python test/debug_step_count.py --data_file <path_to_jsonl>
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
import numpy as np

def count_steps_in_sample(messages, tokenizer):
    """
    Count number of "\n\n" separators in assistant content.
    Returns: number of step boundaries found
    """
    # Get assistant content
    assistant_content = None
    for msg in messages:
        if msg['role'] == 'assistant':
            assistant_content = msg['content']
            break

    if assistant_content is None:
        return 0

    # Count "\n\n" occurrences
    num_separators = assistant_content.count("\n\n")

    return num_separators

def analyze_dataset(data_file, model_name="meta-llama/Llama-3.2-1B-Instruct", max_samples=None):
    """
    Analyze step count statistics in dataset.
    """
    print(f"Loading dataset from: {data_file}")
    dataset = load_dataset('json', data_files=data_file)['train']

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Loaded {len(dataset)} samples")
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Count steps in each sample
    step_counts = []
    examples = []

    for idx, sample in enumerate(dataset):
        messages = sample['messages']
        num_steps = count_steps_in_sample(messages, tokenizer)
        step_counts.append(num_steps)

        # Save first 5 examples for detailed inspection
        if idx < 5:
            assistant_content = None
            for msg in messages:
                if msg['role'] == 'assistant':
                    assistant_content = msg['content']
                    break
            examples.append({
                'idx': idx,
                'num_steps': num_steps,
                'assistant_content': assistant_content
            })

    # Compute statistics
    step_counts = np.array(step_counts)

    print("\n" + "="*80)
    print("STEP COUNT STATISTICS")
    print("="*80)
    print(f"Total samples: {len(step_counts)}")
    print(f"\nStatistics:")
    print(f"  Min steps:    {np.min(step_counts)}")
    print(f"  Max steps:    {np.max(step_counts)}")
    print(f"  Mean steps:   {np.mean(step_counts):.2f}")
    print(f"  Median steps: {np.median(step_counts):.0f}")
    print(f"  Std steps:    {np.std(step_counts):.2f}")

    # Distribution
    print(f"\nDistribution:")
    counter = Counter(step_counts)
    for num_steps in sorted(counter.keys())[:20]:  # Show first 20
        count = counter[num_steps]
        percentage = count / len(step_counts) * 100
        print(f"  {num_steps} steps: {count:4d} samples ({percentage:5.1f}%)")

    if len(counter) > 20:
        print(f"  ... and {len(counter) - 20} more unique step counts")

    # Show examples
    print("\n" + "="*80)
    print("EXAMPLE SAMPLES (First 5)")
    print("="*80)
    for ex in examples:
        print(f"\nSample {ex['idx']}: {ex['num_steps']} steps")
        print(f"Assistant content (first 200 chars):")
        print(f"  {ex['assistant_content'][:200]}...")
        if ex['num_steps'] > 0:
            print(f"  [Contains {ex['num_steps']} '\\n\\n' separators]")

    # Warning if too many steps
    if np.mean(step_counts) > 10:
        print("\n" + "!"*80)
        print("WARNING: Average steps > 10!")
        print(f"This means each sample has {np.mean(step_counts):.1f} '\\n\\n' separators on average.")
        print("For num_prediction_steps=N, this could create many prediction pairs.")
        print("!"*80)

    # Estimate prediction pairs with different num_prediction_steps
    print("\n" + "="*80)
    print("PREDICTION PAIRS ESTIMATION (for batch_size=4)")
    print("="*80)
    for N in [1, 2, 3, 5, 10]:
        # Each sample can have at most min(num_steps, N) prediction pairs
        # For N=2: if sample has 100 steps, it will use min(100-1, 2) = 2 pairs
        total_pairs = 0
        for num_steps in step_counts[:4]:  # First 4 samples (batch_size=4)
            # Number of pairs = min(num_steps, N)
            # But wait, we need at least N+1 boundaries to form N pairs
            if num_steps >= N:
                pairs = N
            else:
                pairs = max(0, num_steps - 1)  # Can form (num_steps - 1) pairs
            total_pairs += pairs

        print(f"  num_prediction_steps={N:2d}: ~{total_pairs} pairs per batch (batch_size=4)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to JSONL data file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to analyze")

    args = parser.parse_args()

    analyze_dataset(args.data_file, args.model_name, args.max_samples)
