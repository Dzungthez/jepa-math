import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def setup_model_and_tokenizer(model_name, debug=0, seed=None, new_tokens=0):
    """Setup model and tokenizer"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    add_token = False
    if new_tokens > 0:
        special_tokens = [f"<|predictor_{i+1}|>" for i in range(new_tokens)]
        special_tokens = [token for token in special_tokens if token not in tokenizer.vocab]

        if len(special_tokens) > 0:
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            if torch.cuda.current_device() == 0:
                print(f"Added {len(special_tokens)} new special tokens")
            add_token = True
        else:
            if torch.cuda.current_device() == 0:
                print(f"No new special tokens to add")
    assert tokenizer.chat_template is not None, f"{model_name} does not have chat template."

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with better device mapping for multi-GPU
    device_map = None
    if torch.cuda.is_available():
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if world_size == 1:
            device_map = "auto"
        else:
            # For multi-GPU with torchrun, don't use device_map
            device_map = None
    
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            # Add these for better multi-GPU stability
            low_cpu_mem_usage=True,
            use_cache=False,  # Disable KV cache for training
        )
    model.resize_token_embeddings(len(tokenizer))
    if add_token:
        if torch.cuda.current_device() == 0:
            print(f"Resized token embeddings to {len(tokenizer)}")

    return model, tokenizer


def save_model(trainer,tokenizer,output_dir):
    if torch.cuda.current_device() == 0:
        trainer.save_model(output_dir)
        # trainer.save_state(output_dir)
        tokenizer.save_pretrained(output_dir)
    if torch.cuda.current_device() == 0:
        print(f"Model saved to {output_dir}")