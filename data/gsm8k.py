from datasets import load_dataset
import json
import torch

def compute_message_spans(messages, tokenizer):
    """
    Compute user and assistant content spans by tokenizing partial conversations.

    Returns:
        user_span: (user_start, user_end) tuple
        assistant_span: (assistant_start, assistant_end) tuple
    """
    # Full conversation
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)['input_ids']

    # System + User only
    user_only_text = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=False
    )
    user_only_ids = tokenizer(user_only_text, add_special_tokens=False)['input_ids']

    # System + User + assistant header
    user_with_prompt_text = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True
    )
    user_with_prompt_ids = tokenizer(user_with_prompt_text, add_special_tokens=False)['input_ids']

    # System only
    system_only_text = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=False
    )
    system_only_ids = tokenizer(system_only_text, add_special_tokens=False)['input_ids']

    # User header marker
    user_header = "<|start_header_id|>user<|end_header_id|>\n\n"
    user_header_ids = tokenizer.encode(user_header, add_special_tokens=False)

    user_content_start = len(system_only_ids) + len(user_header_ids)
    user_content_end = len(user_only_ids) - 1

    assistant_content_start = len(user_with_prompt_ids)
    assistant_content_end = -1  # Will be determined from labels

    return (user_content_start, user_content_end), (assistant_content_start, assistant_content_end)

def load_and_prepare_dataset(data_file, tokenizer,
                             max_length=2048, debug=0, unmask_user=False):
    """Load JSONL dataset and format for training with proper label masking"""
    
    # Load dataset
    dataset = load_dataset('json', data_files=data_file)['train']
    # if  torch.cuda.current_device() == 0:
    print(f"Loaded {len(dataset)} examples from {data_file}")
    
    def tokenize_conversations(examples, unmask_user=False):
        """Tokenize conversations and mask input tokens properly"""
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        user_span_list = []
        assistant_span_list = []

        for msg_idx, messages in enumerate(examples['messages']):
            # Apply chat template if available, otherwise format manually
            formatted_chat = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            
            # Tokenize the formatted conversation with padding to max_length
            tokenized = tokenizer(
                formatted_chat,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
                padding="max_length",  # Pad to max_length for consistent tensor shapes
                return_tensors=None
            )
            
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # Compute spans
            user_span, assistant_span = compute_message_spans(messages, tokenizer)

            # Create labels with proper masking
            labels = create_masked_labels(messages, tokenizer, input_ids, attention_mask, unmask_user)

            # Update assistant_span end from labels
            assistant_end_actual = None
            for i in range(len(labels) - 1, -1, -1):
                if labels[i] != -100:
                    assistant_end_actual = i
                    break
            if assistant_end_actual is not None:
                assistant_span = (assistant_span[0], assistant_end_actual)

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            user_span_list.append(user_span)
            assistant_span_list.append(assistant_span)
        
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list,
            "user_span": user_span_list,
            "assistant_span": assistant_span_list,
        }
    
    def create_masked_labels(messages, tokenizer, input_ids, attention_mask, unmask_user=False):
        """Create labels with input tokens masked (-100)"""
        labels = [-100] * len(input_ids)

        # Mask padding tokens in labels
        for i, mask in enumerate(attention_mask):
            if mask == 0:  # Padding token
                labels[i] = -100

        # Unmask user content if requested
        if unmask_user:
            for msg in messages:
                if msg['role'] == 'user':
                    user_content = msg['content']
                    user_tokens = tokenizer.encode(user_content, add_special_tokens=False)

                    decoded_user = [tokenizer.decode([tok]) for tok in user_tokens]
                    decoded_input = [tokenizer.decode([tok]) for tok in input_ids]

                    # Find and unmask user content
                    for i in range(len(input_ids) - len(user_tokens) + 1):
                        if attention_mask[i] == 1 and decoded_input[i:i+len(user_tokens)] == decoded_user:
                            for j in range(i, min(i + len(user_tokens), len(input_ids))):
                                if attention_mask[j] == 1:
                                    labels[j] = input_ids[j]
                            break

        # Find assistant responses and unmask those tokens (ALWAYS)
        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                
                # Find where this assistant response appears in the tokenized text
                # assistant_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
                assistant_with_eot = assistant_content + tokenizer.eos_token
                assistant_tokens = tokenizer.encode(assistant_with_eot, add_special_tokens=False)
                
                # Find the position of assistant response in input_ids
                decoded_assistant = [tokenizer.decode(item) for item in assistant_tokens]
                decoded_input = [tokenizer.decode(item) for item in input_ids]

                # print(f"decoded_input: {decoded_input}")
                # print(f"decoded_assistant: {decoded_assistant}")
                for i in range(len(input_ids) - len(assistant_tokens) + 1):
                    # Only check non-padding tokens
                    if debug == 4:
                        print(f"=======input_ids: {input_ids[i:i+len(assistant_tokens)]}")
                        print(f"assistant_tokens: {assistant_tokens}")
                    # if attention_mask[i] == 1 and input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                    if attention_mask[i] == 1 and decoded_input[i:i+len(assistant_tokens)] == decoded_assistant:
                        # Unmask the assistant response tokens
                        for j in range(i, min(i + len(assistant_tokens), len(input_ids))):
                            if attention_mask[j] == 1:  # Only unmask non-padding tokens
                                labels[j] = input_ids[j]
                        break
                
                if debug == 4:
                    exit(0)
        # print(f"messages: {messages}")
        # print(f"input_ids: {input_ids}")
        # print(f"attention_mask: {attention_mask}")
        # print(f"labels: {labels}")
        # exit(0)
        return labels
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_conversations(examples, unmask_user=unmask_user),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset