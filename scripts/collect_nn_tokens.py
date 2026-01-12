"""
Collect all token IDs that contain "\n\n" from the dataset.
Run this once offline, save to a JSON file.
"""
import json
from transformers import AutoTokenizer
from collections import Counter

def collect_nn_tokens(dataset_path, tokenizer_name="meta-llama/Llama-3.2-1B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    nn_token_ids = set()
    token_contexts = {}  # For debugging: token_id → example contexts

    # Load dataset
    with open(dataset_path, 'r') as f:
        samples = [json.loads(line) for line in f]

    print(f"Loaded {len(samples)} samples from {dataset_path}")

    for idx, sample in enumerate(samples):
        messages = sample['messages']

        # Apply chat template
        formatted_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize
        tokenized = tokenizer(
            formatted_chat,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors=None
        )

        input_ids = tokenized["input_ids"]

        # Decode each token and check if it contains "\n\n"
        for i, token_id in enumerate(input_ids):
            if token_id == tokenizer.pad_token_id:
                continue

            # Decode this single token
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)

            # Check if token contains "\n\n"
            if '\n\n' in token_text:
                nn_token_ids.add(token_id)

                # Save example context for debugging
                if token_id not in token_contexts:
                    # Get context around this token
                    context_start = max(0, i - 3)
                    context_end = min(len(input_ids), i + 4)
                    context_ids = input_ids[context_start:context_end]
                    context_text = tokenizer.decode(context_ids, skip_special_tokens=False)
                    token_contexts[token_id] = {
                        'token_text': token_text,
                        'example_context': context_text,
                        'count': 1
                    }
                else:
                    token_contexts[token_id]['count'] += 1

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(samples)} samples, found {len(nn_token_ids)} unique tokens")

    # Sort by frequency
    sorted_tokens = sorted(
        token_contexts.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )

    # Print results
    print(f"\n{'='*80}")
    print(f"Found {len(nn_token_ids)} unique token IDs that contain '\\n\\n':")
    print(f"{'='*80}")
    for token_id, info in sorted_tokens[:20]:  # Top 20
        print(f"Token {token_id}: appears {info['count']} times")
        print(f"  Decodes to: {repr(info['token_text'])}")
        print(f"  Example context: {repr(info['example_context'])}")
        print()

    # Save to JSON
    output = {
        'nn_token_ids': list(nn_token_ids),
        'token_details': {
            str(k): v for k, v in token_contexts.items()
        },
        'tokenizer_name': tokenizer_name,
        'dataset_path': dataset_path
    }

    output_path = 'nn_tokens.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(nn_token_ids)} token IDs to {output_path}")
    print(f"Token IDs list: {sorted(nn_token_ids)}")

    return nn_token_ids, token_contexts

if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/gsm8k_step_jepa.jsonl"
    collect_nn_tokens(dataset_path)
