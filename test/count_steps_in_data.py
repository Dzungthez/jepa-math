"""
Count number of steps in dataset by counting "\n\n" occurrences in assistant messages.

This script analyzes the dataset and provides statistics on the number of steps
(double newlines) found in each sample's assistant message.
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import statistics


def count_steps_in_message(message_content: str) -> int:
    """
    Count the number of "\n\n" occurrences in a message.
    
    Args:
        message_content: The text content of the message
    
    Returns:
        Number of "\n\n" occurrences
    """
    return message_content.count("\n\n")


def analyze_dataset(file_path: str):
    """
    Analyze a JSONL dataset file and count steps in assistant messages.
    
    Args:
        file_path: Path to the JSONL file
    """
    print(f"\n{'='*80}")
    print(f"Analyzing dataset: {file_path}")
    print(f"{'='*80}\n")
    
    step_counts = []
    samples_with_details = []
    
    # Read JSONL file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            
            # Find assistant message
            messages = sample.get('messages', [])
            assistant_message = None
            
            for msg in messages:
                if msg.get('role') == 'assistant':
                    assistant_message = msg.get('content', '')
                    break
            
            if assistant_message is None:
                print(f"Warning: No assistant message found in line {line_num}")
                continue
            
            # Count "\n\n" occurrences
            step_count = count_steps_in_message(assistant_message)
            step_counts.append(step_count)
            
            # Store sample details for debugging (first 10 samples)
            if len(samples_with_details) < 10:
                samples_with_details.append({
                    'line_num': line_num,
                    'step_count': step_count,
                    'question': sample.get('question', 'N/A')[:80],
                    'total_steps_metadata': sample.get('total_steps', 'N/A')
                })
    
    # Calculate statistics
    if not step_counts:
        print("Error: No valid samples found in dataset!")
        return
    
    # Calculate percentile function
    def percentile(data, p):
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_data):
            return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
        else:
            return sorted_data[f]
    
    print(f"📊 **Statistics Summary**")
    print(f"-" * 80)
    print(f"Total samples: {len(step_counts)}")
    print(f"Min steps: {min(step_counts)}")
    print(f"Max steps: {max(step_counts)}")
    print(f"Mean steps: {statistics.mean(step_counts):.2f}")
    print(f"Median steps: {statistics.median(step_counts):.2f}")
    print(f"Std deviation: {statistics.stdev(step_counts) if len(step_counts) > 1 else 0:.2f}")
    print(f"25th percentile: {percentile(step_counts, 25):.2f}")
    print(f"75th percentile: {percentile(step_counts, 75):.2f}")
    print()
    
    # Distribution
    print(f"📈 **Step Count Distribution**")
    print(f"-" * 80)
    counter = Counter(step_counts)
    
    for step_count in sorted(counter.keys()):
        count = counter[step_count]
        percentage = (count / len(step_counts)) * 100
        bar = '█' * int(percentage / 2)  # Scale bar to fit screen
        print(f"{step_count:3d} steps: {count:5d} samples ({percentage:5.2f}%) {bar}")
    print()
    
    # Show first 10 samples
    print(f"🔍 **Sample Details (First 10)**")
    print(f"-" * 80)
    for sample in samples_with_details:
        print(f"Line {sample['line_num']:4d}: {sample['step_count']} steps | "
              f"Metadata: {sample['total_steps_metadata']} | "
              f"Question: {sample['question']}...")
    print()
    
    # Comparison with metadata (if available)
    print(f"📋 **Metadata Comparison**")
    print(f"-" * 80)
    
    # Re-read to check metadata
    metadata_available = 0
    metadata_match = 0
    metadata_mismatch_examples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                sample = json.loads(line)
            except:
                continue
            
            if 'total_steps' in sample:
                metadata_available += 1
                
                # Find assistant message again
                messages = sample.get('messages', [])
                assistant_message = None
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        assistant_message = msg.get('content', '')
                        break
                
                if assistant_message:
                    counted_steps = count_steps_in_message(assistant_message)
                    metadata_steps = sample['total_steps']
                    
                    if counted_steps == metadata_steps:
                        metadata_match += 1
                    elif len(metadata_mismatch_examples) < 5:
                        metadata_mismatch_examples.append({
                            'line_num': line_num,
                            'counted': counted_steps,
                            'metadata': metadata_steps,
                            'question': sample.get('question', 'N/A')[:60]
                        })
    
    if metadata_available > 0:
        print(f"Samples with 'total_steps' metadata: {metadata_available}")
        print(f"Metadata matches counted steps: {metadata_match} ({(metadata_match/metadata_available)*100:.2f}%)")
        print(f"Metadata mismatches: {metadata_available - metadata_match}")
        
        if metadata_mismatch_examples:
            print(f"\n⚠️  **Mismatch Examples (First 5)**")
            print(f"-" * 80)
            for ex in metadata_mismatch_examples:
                print(f"Line {ex['line_num']:4d}: Counted={ex['counted']}, Metadata={ex['metadata']} | "
                      f"Q: {ex['question']}...")
    else:
        print("No 'total_steps' metadata found in dataset.")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Count steps (\\n\\n occurrences) in dataset assistant messages",
        epilog="""
Examples:
  # Analyze default dataset
  python count_steps_in_data.py
  
  # Analyze specific dataset
  python count_steps_in_data.py ../datasets/gsm8k_train.jsonl
  
  # Analyze multiple datasets
  python count_steps_in_data.py ../datasets/*.jsonl
        """
    )
    parser.add_argument(
        "file_paths",
        type=str,
        nargs='*',
        default=["../datasets/gsm8k_step_jepa.jsonl"],
        help="Path(s) to JSONL dataset file(s) (default: ../datasets/gsm8k_step_jepa.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    
    # Process each file
    for file_path_str in args.file_paths:
        file_path = script_dir / file_path_str
        
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            print(f"Skipping...\n")
            continue
        
        analyze_dataset(str(file_path))
        print()  # Extra newline between datasets


if __name__ == "__main__":
    main()

