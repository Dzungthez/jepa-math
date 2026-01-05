import json
import re
import torch
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# Configuration parameters
DEFAULT_MODEL_PATH = "./checkpoints_sft/checkpoint-1287"
DEFAULT_TEST_FILE = "datasets/gsm8k_test.jsonl"
DEFAULT_OUTPUT_FILE = "./evaluation_results.jsonl"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0  # Greedy decoding


def extract_boxed_answer(text):
    """Extract answer from \\boxed{} format (used in DeepSeek system prompt)"""
    pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(pattern, text)
    if match:
        answer = match.group(1).strip()
        return normalize_answer(answer)
    return None


def extract_hash_answer(text):
    """Extract answer from #### format (GSM8K standard format)"""
    pattern = r'\n#### (.+)$'
    match = re.search(pattern, text)
    if match:
        answer = match.group(1).strip()
        return normalize_answer(answer)
    return None


def extract_final_number(text):
    """Try to extract the last number from the text as a fallback"""
    numbers = re.findall(r'[-+]?(?:\d*\.*\d+)', text)
    if numbers:
        return normalize_answer(numbers[-1])
    return None


def normalize_answer(answer):
    """Normalize answer for comparison"""
    if answer is None:
        return None
    
    # Remove common text patterns
    answer = answer.replace('$', '').replace(',', '').strip()
    
    # Try to convert to number and normalize
    try:
        num = float(answer)
        # If it's a whole number, return as int
        if num.is_integer():
            return str(int(num))
        else:
            # Round to reasonable precision
            return f"{num:.10g}"
    except (ValueError, TypeError):
        # If not a number, return cleaned string
        return answer.strip()


def extract_answer_from_generated(generated_text):
    """Extract answer from generated text - try multiple formats"""
    # Try boxed format first (from DeepSeek system prompt)
    answer = extract_boxed_answer(generated_text)
    if answer is not None:
        return answer
    
    # Try #### format (GSM8K standard)
    answer = extract_hash_answer(generated_text)
    if answer is not None:
        return answer
    
    # Fallback: try to extract last number
    answer = extract_final_number(generated_text)
    return answer


def eval_gsm8k(generated, ground_truth):
    """
    Evaluate GSM8K answer.
    
    Args:
        generated: Generated response text
        ground_truth: Ground truth in GSM8K format (with ####)
    
    Returns:
        (is_correct, gt_answer, gen_answer)
    """
    # Extract ground truth answer
    gt_answer = extract_hash_answer(ground_truth)
    
    # Extract generated answer
    gen_answer = extract_answer_from_generated(generated)
    
    # Compare
    is_correct = (gt_answer is not None and 
                  gen_answer is not None and 
                  gt_answer == gen_answer)
    
    return is_correct, gt_answer, gen_answer

def load_test_data(test_file, max_examples=None):
    """Load test data from JSONL file"""
    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
            if max_examples and len(test_data) >= max_examples:
                break
    return test_data


def load_model(model_path):
    """Load tokenizer and model with vLLM"""
    model_path = Path(model_path)
    print(f"Loading model with vLLM from: {model_path}")
    print("=" * 80)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with vLLM for fast inference
    print("Loading model with vLLM for optimized inference...")
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=1,  # Adjust based on number of GPUs
        gpu_memory_utilization=0.9,
        max_model_len=4096,  # Adjust based on your needs
        trust_remote_code=True
    )
    
    print(f"\n✓ Model loaded successfully with vLLM")
    print("  vLLM provides highly optimized inference with PagedAttention")
    print("=" * 80)
    
    return llm, tokenizer


def prepare_prompts(test_data, tokenizer):
    """Prepare all prompts from test data"""
    all_prompts = []
    ground_truths = []
    questions = []
    new_system_prompt = "Please solve the problem step by step (separate steps with double newlines), but keep it short and put your final answer (do not include any other text or units) within \\boxed{}."
    for example in test_data:
        messages = example["messages"]
        ground_truth = messages[-1]["content"]
        input_messages = messages[:-1]  # Exclude assistant's answer
        input_messages[0]["content"] = new_system_prompt
        
        prompt = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        all_prompts.append(prompt)
        ground_truths.append(ground_truth)
        questions.append(messages[1]["content"])
    
    return all_prompts, ground_truths, questions


def evaluate_batch(llm, all_prompts, ground_truths, questions, output_file, 
                   max_new_tokens=1536, temperature=0.0):
    """Run batch evaluation and save results"""
    # Batch generate with vLLM
    print(f"Batch generating {len(all_prompts)} responses with vLLM...")
    sampling_params = SamplingParams(
        temperature=temperature if temperature > 0 else 0.0,
        max_tokens=max_new_tokens,
        top_p=1.0 if temperature == 0 else 0.95,
    )
    outputs = llm.generate(all_prompts, sampling_params)
    print(f"✓ Generation complete\n")
    
    # Evaluate and save results
    results = []
    correct_count = 0
    total_count = 0
    
    with open(output_file, 'w') as f:
        for idx, (output, ground_truth, question) in enumerate(tqdm(
            zip(outputs, ground_truths, questions), 
            total=len(outputs),
            desc="Evaluating"
        )):
            try:
                # Extract generated text
                generated_response = output.outputs[0].text.strip()
                
                # Evaluate
                is_correct, gt_answer, gen_answer = eval_gsm8k(generated_response, ground_truth)
                
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                # Compute accuracy so far
                accuracy = correct_count / total_count * 100
                
                # Create result entry
                result = {
                    "index": idx,
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_response": generated_response,
                    "gt_answer": gt_answer,
                    "gen_answer": gen_answer,
                    "correct": is_correct,
                    "accuracy_so_far": accuracy
                }
                results.append(result)
                
                # Write to file
                f.write(json.dumps(result) + '\n')
                f.flush()
                
                # Print progress every 10 examples
                if (idx + 1) % 10 == 0:
                    print(f"After {idx + 1} examples: {correct_count}/{total_count} correct ({accuracy:.2f}%)")
                
            except Exception as e:
                print(f"\n❌ Error at index {idx}: {e}")
                result = {
                    "index": idx,
                    "question": question,
                    "error": str(e),
                    "correct": False
                }
                results.append(result)
                f.write(json.dumps(result) + '\n')
                f.flush()
    
    return results, correct_count, total_count


def save_summary(model_path, test_file, total_count, correct_count, 
                 output_file, max_new_tokens, temperature):
    """Save evaluation summary"""
    summary = {
        "model_path": model_path,
        "test_file": test_file,
        "total_examples": total_count,
        "correct": correct_count,
        "accuracy": correct_count / total_count if total_count > 0 else 0,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }
    
    summary_file = output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary_file


def main(model_path=None, test_file=None, output_file=None, 
         max_new_tokens=None, temperature=None, max_examples=None):
    """Main evaluation function"""
    # Set defaults
    model_path = model_path or DEFAULT_MODEL_PATH
    test_file = test_file or DEFAULT_TEST_FILE
    output_file = output_file or DEFAULT_OUTPUT_FILE
    max_new_tokens = max_new_tokens or DEFAULT_MAX_NEW_TOKENS
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
    
    # Print configuration
    print("✓ All libraries imported successfully")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    print("=" * 80)
    print("Step-JEPA Evaluation Configuration")
    print("=" * 80)
    print(f"Model Path: {model_path}")
    print(f"Test File: {test_file}")
    print(f"Output File: {output_file}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Max Examples: {max_examples}")
    print("=" * 80)
    
    # Load test data
    print(f"Loading test data from: {test_file}")
    test_data = load_test_data(test_file, max_examples)
    print(f"✓ Loaded {len(test_data)} examples")
    print("=" * 80)
    
    # Load model
    llm, tokenizer = load_model(model_path)
    
    # Prepare prompts
    print(f"Running batch evaluation on {len(test_data)} examples...")
    print("=" * 80)
    all_prompts, ground_truths, questions = prepare_prompts(test_data, tokenizer)
    
    # Evaluate
    results, correct_count, total_count = evaluate_batch(
        llm, all_prompts, ground_truths, questions, output_file,
        max_new_tokens=max_new_tokens, temperature=temperature
    )
    
    # Print final results
    print("\n" + "=" * 80)
    print("✓ Batch evaluation complete!")
    print(f"Results saved to: {output_file}")
    print("=" * 80)
    
    print("=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total examples: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {total_count - correct_count}")
    print(f"Accuracy: {correct_count / total_count * 100:.2f}%")
    print("=" * 80)
    
    # Save summary
    summary_file = save_summary(
        model_path, test_file, total_count, correct_count,
        output_file, max_new_tokens, temperature
    )
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K test set")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Path to test JSONL file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to output JSONL file")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to evaluate")
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        test_file=args.test_file,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_examples=args.max_examples
    )