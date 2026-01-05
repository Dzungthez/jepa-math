import json
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


tokenizer.pad_token = tokenizer.eos_token
# Load a few samples
with open('../datasets/gsm8k_step_jepa.jsonl', 'r') as f:
    samples = []
    for i, line in enumerate(f):
        if i >= 10:  # Test 10 samples
            break
        samples.append(json.loads(line))

def find_step_boundaries(input_ids, processing_class, labels=None):
    """Same logic as in jepa_trainer.py - only searches within assistant content"""
    sep_tokens = processing_class.encode("\n\n", add_special_tokens=False)
    if isinstance(input_ids, list):
        ids_list = input_ids
    else:
        ids_list = input_ids.tolist()
    
    # Find the start of assistant content (where labels != -100)
    assistant_start = 0
    if labels is not None:
        labels_list = labels if isinstance(labels, list) else labels.tolist()
        # Find first position where label != -100 (assistant content starts)
        for i, label in enumerate(labels_list):
            if label != -100:
                assistant_start = i
                break
    
    # Only search for \n\n within assistant content
    search_start = assistant_start
    search_end = len(ids_list)  # Full length, not limited by sep_tokens
    
    # Debug print
    print(f"    DEBUG find_step_boundaries: assistant_start={assistant_start}, search_range=[{search_start}, {search_end})")
    
    # Use decode method to find \n\n positions - more reliable than token search
    # because \n\n can be tokenized differently (271 standalone, 382 as '.\n\n', etc.)
    decode_limit = min(search_start + 700, search_end)
    assistant_text = processing_class.decode(ids_list[search_start:decode_limit], skip_special_tokens=False)
    
    # Find first \n\n in assistant content
    first_nn_pos = assistant_text.find('\n\n')
    if first_nn_pos == -1:
        print(f"    DEBUG: No \\n\\n found in assistant content")
        return None, None
    
    # Map first \n\n text position back to token position
    step1_end = None
    for i in range(search_start, decode_limit):
        decoded_so_far = processing_class.decode(ids_list[search_start:i+1], skip_special_tokens=False)
        if len(decoded_so_far) >= first_nn_pos + 2 and decoded_so_far[first_nn_pos:first_nn_pos+2] == '\n\n':
            step1_end = i
            print(f"    DEBUG: Found first \\n\\n via decode method at token {i} (text pos {first_nn_pos})")
            break
    
    if step1_end is None:
        print(f"    DEBUG: Could not map first \\n\\n to token position")
        return None, None
    
    # Find second \n\n in assistant content (must be after first one)
    second_nn_pos = assistant_text.find('\n\n', first_nn_pos + 2)
    if second_nn_pos == -1:
        print(f"    DEBUG: Second \\n\\n not found")
        return None, None
    
    # Map second \n\n text position back to token position
    step2_end = None
    for i in range(step1_end + 1, decode_limit):  # Must be after step1_end
        decoded_so_far = processing_class.decode(ids_list[search_start:i+1], skip_special_tokens=False)
        if len(decoded_so_far) >= second_nn_pos + 2 and decoded_so_far[second_nn_pos:second_nn_pos+2] == '\n\n':
            step2_end = i
            print(f"    DEBUG: Found second \\n\\n via decode method at token {i} (text pos {second_nn_pos})")
            break
    
    if step2_end is None:
        print(f"    DEBUG: Could not map second \\n\\n to token position")
        return None, None
    
    # Sanity check: step2_end must be > step1_end
    if step2_end <= step1_end:
        print(f"    DEBUG: ERROR - step2_end ({step2_end}) <= step1_end ({step1_end}), invalid!")
        return None, None
    
    return step1_end, step2_end

def create_masked_labels(messages, tokenizer, input_ids, attention_mask, debug=False):
    """Create labels with input tokens masked (-100) - same as in gsm8k.py"""
    labels = [-100] * len(input_ids)
    
    # Mask padding tokens in labels
    for i, mask in enumerate(attention_mask):
        if mask == 0:  # Padding token
            labels[i] = -100
    
    # Find assistant responses and unmask only those tokens
    found_assistant = False
    for msg in messages:
        if msg['role'] == 'assistant':
            assistant_content = msg['content']
            
            # Find where this assistant response appears in the tokenized text
            assistant_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
            
            # Find the position of assistant response in input_ids
            decoded_assistant = [tokenizer.decode(item) for item in assistant_tokens]
            decoded_input = [tokenizer.decode(item) for item in input_ids]
            
            if debug:
                print(f"      DEBUG create_masked_labels: assistant_tokens length={len(assistant_tokens)}")
                print(f"      DEBUG: First 5 decoded_assistant: {decoded_assistant[:5]}")
                print(f"      DEBUG: First 5 decoded_input: {decoded_input[:5]}")
            
            for i in range(len(input_ids) - len(assistant_tokens) + 1):
                # Only check non-padding tokens
                if attention_mask[i] == 1 and decoded_input[i:i+len(assistant_tokens)] == decoded_assistant:
                    # Unmask the assistant response tokens
                    for j in range(i, min(i + len(assistant_tokens), len(input_ids))):
                        if attention_mask[j] == 1:  # Only unmask non-padding tokens
                            labels[j] = input_ids[j]
                    found_assistant = True
                    if debug:
                        print(f"      DEBUG: Found assistant at position {i}")
                    break
            
            if not found_assistant and debug:
                print(f"      DEBUG: Could not find assistant content in tokenized sequence!")
                print(f"      DEBUG: Trying to find partial match...")
                # Try to find partial match
                for i in range(len(input_ids) - 10):
                    if attention_mask[i] == 1:
                        partial_match = decoded_input[i:i+10] == decoded_assistant[:10]
                        if partial_match:
                            print(f"      DEBUG: Found partial match at {i}")
            break
    
    return labels

# Test each sample
print("=" * 80)
print("Testing step boundaries detection")
print("=" * 80)

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
    attention_mask = tokenized["attention_mask"]
    
    # Create labels (same as in gsm8k.py)
    labels = create_masked_labels(messages, tokenizer, input_ids, attention_mask, debug=(idx == 0))
    
    # Debug: Check labels
    num_unmasked = sum(1 for l in labels if l != -100)
    assistant_start_from_labels = next((i for i, l in enumerate(labels) if l != -100), None)
    print(f"\nSample {idx}:")
    print(f"  Total tokens: {len(input_ids)}, Unmasked labels: {num_unmasked}, Assistant starts at: {assistant_start_from_labels}")
    
    # Debug: Check how \n\n is tokenized in different contexts
    sep_tokens_standalone = tokenizer.encode("\n\n", add_special_tokens=False)
    sep_tokens_after_word = tokenizer.encode("word\n\n", add_special_tokens=False)
    sep_tokens_before_word = tokenizer.encode("\n\nword", add_special_tokens=False)
    print(f"  \\n\\n standalone tokenizes to: {sep_tokens_standalone}")
    print(f"  'word\\n\\n' tokenizes to: {sep_tokens_after_word}")
    print(f"  '\\n\\nword' tokenizes to: {sep_tokens_before_word}")
    
    # Use the standalone version
    sep_tokens = sep_tokens_standalone
    
    # Debug: Check if \n\n exists in full sequence
    ids_list = input_ids if isinstance(input_ids, list) else input_ids.tolist()
    found_in_full = False
    for i in range(len(ids_list) - len(sep_tokens) + 1):
        if ids_list[i:i+len(sep_tokens)] == sep_tokens:
            found_in_full = True
            print(f"  Found \\n\\n at position {i} in FULL sequence")
            break
    if not found_in_full:
        print(f"  WARNING: No \\n\\n found in FULL sequence!")
    
    # Debug: Check if token 271 exists in assistant content
    assistant_start_from_labels = next((i for i, l in enumerate(labels) if l != -100), None)
    if assistant_start_from_labels is not None:
        assistant_ids = ids_list[assistant_start_from_labels:assistant_start_from_labels+100]
        token_271_count = assistant_ids.count(271)
        token_382_count = assistant_ids.count(382)  # Check if 382 is related to \n\n
        print(f"  Token 271 appears {token_271_count} times in first 100 tokens of assistant content")
        print(f"  Token 382 appears {token_382_count} times in first 100 tokens of assistant content")
        
        # Check what token 382 decodes to
        if token_382_count > 0:
            token_382_text = tokenizer.decode([382], skip_special_tokens=False)
            print(f"  Token 382 decodes to: {repr(token_382_text)}")
            
            # Check if 382 appears near \n\n positions
            for i in range(assistant_start_from_labels, min(assistant_start_from_labels+200, len(ids_list))):
                if ids_list[i] == 382:
                    # Decode context around it
                    context = tokenizer.decode(ids_list[max(0, i-3):i+4], skip_special_tokens=False)
                    if '\n\n' in context:
                        print(f"    Found token 382 at position {i}, context: {repr(context)}")
                        break
        
        # Try to find \n\n in assistant content by decoding and searching
        assistant_text = tokenizer.decode(ids_list[assistant_start_from_labels:assistant_start_from_labels+200], skip_special_tokens=False)
        nn_positions_in_text = []
        pos = 0
        while True:
            pos = assistant_text.find('\n\n', pos)
            if pos == -1:
                break
            nn_positions_in_text.append(pos)
            pos += 2
        
        if nn_positions_in_text:
            print(f"  Found {len(nn_positions_in_text)} \\n\\n in decoded assistant text at positions: {nn_positions_in_text[:5]}")
            # Try to find which tokens correspond to first \n\n
            for nn_pos in nn_positions_in_text[:2]:
                # Find token position by decoding incrementally
                for i in range(assistant_start_from_labels, min(assistant_start_from_labels+200, len(ids_list))):
                    decoded_so_far = tokenizer.decode(ids_list[assistant_start_from_labels:i+1], skip_special_tokens=False)
                    if len(decoded_so_far) >= nn_pos + 2 and decoded_so_far[nn_pos:nn_pos+2] == '\n\n':
                        print(f"    First \\n\\n at text pos {nn_pos} corresponds to token position {i}")
                        # Check what tokens are at position i
                        print(f"    Tokens around {i}: {ids_list[max(0, i-2):i+3]}")
                        break
        else:
            print(f"  WARNING: No \\n\\n found in decoded assistant text!")
    
    # Find step boundaries (only in assistant content)
    step1_end, step2_end = find_step_boundaries(input_ids, tokenizer, labels=labels)
    
    # Decode around step1_end and step2_end to see what we're detecting
    print(f"  step1_end: {step1_end}, step2_end: {step2_end}")
    
    if step1_end is not None:
        # Show context around step1_end
        start = max(0, step1_end - 5)
        end = min(len(input_ids), step1_end + 10)
        context1 = tokenizer.decode(input_ids[start:end], skip_special_tokens=False)
        print(f"  Context around step1_end:")
        print(f"    {repr(context1)}")
        
        # Check if it's in system/user or assistant (using labels)
        assistant_start = 0
        for i, label in enumerate(labels):
            if label != -100:
                assistant_start = i
                break
        print(f"  Assistant content starts at token position: {assistant_start} (from labels)")
                print(f"  step1_end ({step1_end}) is {'INSIDE' if step1_end >= assistant_start else 'OUTSIDE'} assistant content")
    
    if step2_end is not None:
        start = max(0, step2_end - 5)
        end = min(len(input_ids), step2_end + 10)
        context2 = tokenizer.decode(input_ids[start:end], skip_special_tokens=False)
        print(f"  Context around step2_end:")
        print(f"    {repr(context2)}")

# Now test with a batch (simulate what happens in training)
print("\n" + "=" * 80)
print("Testing with batch (simulating training)")
print("=" * 80)

batch_samples = samples[:4]  # First 4 samples as a batch
step1_positions = []
step2_positions = []

for batch_idx, sample in enumerate(batch_samples):
    print(f"Batch sample {batch_idx}:")
    messages = sample['messages']
    formatted_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = tokenizer(
        formatted_chat,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors=None
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # Create labels
    labels = create_masked_labels(messages, tokenizer, input_ids, attention_mask, debug=(batch_idx == 0))
    
    # Debug labels
    num_unmasked = sum(1 for l in labels if l != -100)
    assistant_start = next((i for i, l in enumerate(labels) if l != -100), None)
    print(f"  Unmasked labels: {num_unmasked}, Assistant starts at: {assistant_start}")
    
    # Find step boundaries (only in assistant content)
    step1_end, step2_end = find_step_boundaries(input_ids, tokenizer, labels=labels)
    print(f"  step1_end: {step1_end}, step2_end: {step2_end}")
    step1_positions.append(step1_end)
    step2_positions.append(step2_end)

print(f"\nStep1 positions in batch: {step1_positions}")
print(f"Step2 positions in batch: {step2_positions}")
print(f"All step1 same? {all(x == step1_positions[0] for x in step1_positions)}")
print(f"All step2 same? {all(x == step2_positions[0] for x in step2_positions)}")

