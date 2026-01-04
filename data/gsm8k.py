from datasets import load_dataset
import json
import torch

def load_and_prepare_dataset(data_file, tokenizer,
                             max_length=2048, debug=0):
    """Load JSONL dataset and format for training with proper label masking"""
    
    # Load dataset
    dataset = load_dataset('json', data_files=data_file)['train']
    if  torch.cuda.current_device() == 0:
        print(f"Loaded {len(dataset)} examples from {data_file}")
    
    def tokenize_conversations(examples):
        """Tokenize conversations and mask input tokens properly"""
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

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
                padding="max_length",  # Pad to max_length for consistent tensor shapes
                return_tensors=None
            )
            
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            # Create labels with proper masking

            labels = create_masked_labels(messages, tokenizer, input_ids, attention_mask)
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
        
            return {
                "input_ids": input_ids_list,
                "labels": labels_list,
                "attention_mask": attention_mask_list,
            }
    
    def create_masked_labels(messages, tokenizer, input_ids, attention_mask):
        """Create labels with input tokens masked (-100)"""
        labels = [-100] * len(input_ids)
        
        # Mask padding tokens in labels
        for i, mask in enumerate(attention_mask):
            if mask == 0:  # Padding token
                labels[i] = -100
        
        # Find assistant responses and unmask only those tokens
        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                
                # Find where this assistant response appears in the tokenized text
                assistant_tokens = tokenizer.encode(assistant_content, add_special_tokens=False)
                
                # Find the position of assistant response in input_ids
                decoded_assistant = [tokenizer.decode(item) for item in assistant_tokens]
                decoded_input = [tokenizer.decode(item) for item in input_ids]
                for i in range(len(input_ids) - len(assistant_tokens) + 1):
                    # Only check non-padding tokens
                    if debug == 4 and torch.cuda.current_device() == 0:
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
        
        return labels
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_conversations,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset