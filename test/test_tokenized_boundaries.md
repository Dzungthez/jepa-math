# Test Step Boundaries After Tokenization

Vấn đề: Trong raw text, các vị trí `\n\n` khác nhau, nhưng sau khi tokenize có thể giống nhau.

## Cần kiểm tra:

1. Sau khi apply chat template và tokenize, vị trí `\n\n` đầu tiên và thứ hai có còn khác nhau không?
2. Có phải do padding làm cho các vị trí này align lại không?
3. Có phải `\n\n` đầu tiên nằm trong phần system/user message không?

## Script để test (chạy trong notebook hoặc môi trường có transformers):

```python
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Load samples
with open('../datasets/gsm8k_step_jepa.jsonl', 'r') as f:
    samples = [json.loads(line) for i, line in enumerate(f) if i < 10]

def find_step_boundaries(input_ids, processing_class):
    sep_tokens = processing_class.encode("\n\n", add_special_tokens=False)
    ids_list = input_ids.tolist() if hasattr(input_ids, 'tolist') else input_ids
    
    step1_end = None
    for i in range(len(ids_list) - len(sep_tokens) + 1):
        if ids_list[i:i+len(sep_tokens)] == sep_tokens:
            step1_end = i + len(sep_tokens) - 1
            break
    
    if step1_end is None:
        return None, None
    
    step2_end = None
    for i in range(step1_end + len(sep_tokens), len(ids_list) - len(sep_tokens) + 1):
        if ids_list[i:i+len(sep_tokens)] == sep_tokens:
            step2_end = i + len(sep_tokens) - 1
            break
    
    return step1_end, step2_end

# Test với tokenization
step1_positions = []
step2_positions = []

for idx, sample in enumerate(samples):
    messages = sample['messages']
    
    # Apply chat template (same as in gsm8k.py)
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
    step1_end, step2_end = find_step_boundaries(input_ids, tokenizer)
    
    step1_positions.append(step1_end)
    step2_positions.append(step2_end)
    
    # Check if step1 is in system/user or assistant
    # Find where assistant starts
    assistant_msg = None
    for msg in messages:
        if msg['role'] == 'assistant':
            assistant_msg = msg
            break
    
    if assistant_msg:
        before_assistant = []
        for msg in messages:
            if msg['role'] == 'assistant':
                break
            before_assistant.append(msg)
        
        if len(before_assistant) > 0:
            formatted_before = tokenizer.apply_chat_template(
                before_assistant,
                tokenize=False,
                add_generation_prompt=False,
            )
            before_tokens = tokenizer.encode(formatted_before, add_special_tokens=False)
            assistant_start = len(before_tokens)
            
            print(f"Sample {idx}: step1_end={step1_end}, assistant_start={assistant_start}, "
                  f"step1_in_assistant={step1_end >= assistant_start if step1_end else False}")

print(f"\nStep1 positions: {step1_positions}")
print(f"Step2 positions: {step2_positions}")
print(f"All step1 same? {all(x == step1_positions[0] for x in step1_positions if x is not None)}")
print(f"All step2 same? {all(x == step2_positions[0] for x in step2_positions if x is not None)}")
```

## Kết quả mong đợi:

Nếu tất cả `step1_end` giống nhau sau khi tokenize, có thể:
1. `\n\n` đầu tiên nằm trong phần system/user (cùng vị trí sau tokenize)
2. Hoặc do padding làm align các vị trí

