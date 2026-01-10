"""
Debug script to simulate _build_triple_batch() and verify num_pairs calculation.
Usage: python test/debug_triple_batch.py --data_file <path> --batch_size 4 --num_prediction_steps 2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from data.gsm8k import load_and_prepare_dataset

def find_step_boundaries(input_ids, labels, tokenizer, assistant_span, num_steps=2):
    """
    Simulate _find_step_boundaries() logic.
    """
    ids = input_ids.tolist()

    if assistant_span is not None:
        assistant_start, assistant_end = assistant_span
    else:
        # Fallback
        assistant_start = None
        for i, x in enumerate(labels.tolist()):
            if x != -100:
                assistant_start = i
                break
        if assistant_start is None:
            return None

        assistant_end = None
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] != -100:
                assistant_end = i
                break
        if assistant_end is None:
            return None

    # Search for "\n\n" within assistant span
    sep_tokens = tokenizer.encode("\n\n", add_special_tokens=False)
    occ = []

    for i in range(assistant_start, assistant_end - len(sep_tokens) + 2):
        if ids[i:i+len(sep_tokens)] == sep_tokens:
            occ.append(i + len(sep_tokens) - 1)
            # CRITICAL: break when we have enough
            if len(occ) >= num_steps:
                break

    if len(occ) < 2:
        return None

    return occ

def simulate_triple_batch(data_file, model_name, batch_size=4, num_prediction_steps=2, max_length=1024):
    """
    Simulate the triple batch construction and count pairs.
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    print(f"Loading dataset from: {data_file}")
    dataset = load_and_prepare_dataset(
        data_file,
        tokenizer,
        max_length=max_length,
        debug=0,
        unmask_user=False
    )

    print(f"Total samples in dataset: {len(dataset)}")
    print(f"\nSimulating triple batch with batch_size={batch_size}, num_prediction_steps={num_prediction_steps}")
    print("="*80)

    # Take first batch
    batch = dataset.select(range(min(batch_size, len(dataset))))

    print(f"\nProcessing {len(batch)} samples...")

    all_step_boundaries = []
    num_pairs_per_sample = []

    for i in range(len(batch)):
        sample = batch[i]
        input_ids = torch.tensor(sample['input_ids'])
        labels = torch.tensor(sample['labels'])

        # Get metadata
        user_span = None
        assistant_span = None
        if 'user_span' in sample:
            user_span = (sample['user_span'][0], sample['user_span'][1])
        if 'assistant_span' in sample:
            assistant_span = (sample['assistant_span'][0], sample['assistant_span'][1])

        # Find boundaries
        boundaries = find_step_boundaries(
            input_ids,
            labels,
            tokenizer,
            assistant_span,
            num_steps=num_prediction_steps + 1
        )

        if boundaries is None or len(boundaries) < 2:
            # Fallback
            last_token = None
            for idx in range(len(input_ids) - 1, -1, -1):
                if input_ids[idx] != tokenizer.pad_token_id:
                    last_token = idx
                    break

            if last_token is None:
                last_token = len(input_ids) - 1

            num_parts = num_prediction_steps + 1
            boundaries = [last_token * (j+1) // (num_parts + 1)
                         for j in range(num_prediction_steps + 1)]
            print(f"  Sample {i}: Used FALLBACK, created {len(boundaries)} boundaries")
        else:
            print(f"  Sample {i}: Found {len(boundaries)} boundaries from '\\n\\n' search")

        num_pairs = len(boundaries) - 1

        all_step_boundaries.append(boundaries)
        num_pairs_per_sample.append(num_pairs)

        print(f"    → num_pairs = {num_pairs}")
        if num_pairs > num_prediction_steps:
            print(f"    ⚠️  WARNING: num_pairs ({num_pairs}) > num_prediction_steps ({num_prediction_steps})!")

    # Compute total
    total_pairs = sum(num_pairs_per_sample)
    avg_pairs = total_pairs / len(num_pairs_per_sample)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Batch size: {len(batch)}")
    print(f"num_prediction_steps: {num_prediction_steps}")
    print(f"num_pairs_per_sample: {num_pairs_per_sample}")
    print(f"Total pairs in batch: {total_pairs}")
    print(f"Average pairs per sample: {avg_pairs:.2f}")
    print(f"\nExpected pairs per sample: ~{num_prediction_steps} (if enough steps)")
    print(f"Expected total pairs: ~{batch_size * num_prediction_steps}")

    if total_pairs > batch_size * num_prediction_steps * 2:
        print("\n" + "!"*80)
        print("🚨 ANOMALY DETECTED!")
        print(f"Total pairs ({total_pairs}) is much larger than expected ({batch_size * num_prediction_steps})")
        print("This suggests a bug in the boundary finding logic.")
        print("!"*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_prediction_steps", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)

    args = parser.parse_args()

    simulate_triple_batch(
        args.data_file,
        args.model_name,
        args.batch_size,
        args.num_prediction_steps,
        args.max_length
    )
