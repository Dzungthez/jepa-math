import sys
import os
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling
import argparse
import shutil
import math
from utils import setup_model_and_tokenizer
from data.gsm8k import load_and_prepare_dataset

from transformers import Trainer
from jepa_trainer import JepaTrainer

def main():
    parser = argparse.ArgumentParser(description="Step-JEPA Training (Adapted from finetune.py)")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--eval_file", type=str, help="Path to evaluation JSONL file")
    parser.add_argument("--full_data_file", type=str, help="Path to full data JSONL file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name/path")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_step_jepa_adapted", help="Output directory")
    
    # Training arguments
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation steps")
    
    # Step-JEPA arguments
    parser.add_argument("--step_jepa", action="store_true", help="Enable Step-JEPA mode (isolate Step 2)")
    parser.add_argument("--regular", action="store_true", help="Use regular trainer (SFT) without JEPA")
    parser.add_argument("--predictors", type=int, default=1, help="Number of K predictor tokens after Step 1")
    parser.add_argument("--lbd", type=float, default=0.1, help="Lambda for JEPA loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma for LM loss")
    parser.add_argument("--last_token", type=int, default=-1, help="Index of last token for embedding extraction")
    parser.add_argument("--jepa_l2", action="store_true", help="Use L2 norm as JEPA loss")
    parser.add_argument("--jepa_mse", action="store_true", help="Use MSE as JEPA loss")
    parser.add_argument("--infonce", action="store_true", help="Use InfoNCE loss")
    parser.add_argument("--jepa_ratio", type=float, default=-1.0, help="Random JEPA loss dropout ratio")
    parser.add_argument("--additive_mask", type = bool, default=True, help="Use additive mask")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", type=int, default=5, help="Debug level")
    parser.add_argument("--pretrain", action="store_true", help="Pretraining mode")
    parser.add_argument("--same_flop", action="store_true", help="Adjust epochs/steps to match FLOPs")
    
    args = parser.parse_args()
    if torch.cuda.is_available():
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if world_size == 1:
            device_map = "auto"
        else:
            # For multi-GPU with torchrun, don't use device_map            
            world_size = 1
    if torch.cuda.current_device() == 0:
        print(f"Using {world_size} GPUs")
    print("="*80)
    print("Step-JEPA Training (Adapted from finetune.py)")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Mode: {'Regular (No JEPA)' if args.regular else 'Step-JEPA' if args.step_jepa else 'Base RepresentationTrainer'}")
    if not args.regular:
        print(f"Lambda (JEPA): {args.lbd}")
        print(f"Gamma (LM): {args.gamma}")
        print(f"Last token: {args.last_token}")
        print(f"Predictors (K): {args.predictors}")
        if args.step_jepa:
            print(f"Step-JEPA: Isolate Step 2, K={args.predictors} tokens after Step 1")

    print("="*80)

    print("\n1. Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        debug=args.debug,
        seed=args.seed,
        new_tokens=args.predictors
    )

    print("\n2. Loading dataset...")
    if args.train_file is not None:
        train_dataset = load_and_prepare_dataset(
            args.train_file,
            tokenizer,
            max_length=args.max_length,
            debug=args.debug
        )
        if args.eval_file is not None:
            eval_dataset = load_and_prepare_dataset(
                args.eval_file,
                tokenizer,
                max_length=args.max_length,
                debug=args.debug
            )
        else:
            eval_dataset = None
            if torch.cuda.current_device() == 0:
                print("No evaluation file provided")
    else:
        # load fulldata and split into train and eval
        full_dataset = load_and_prepare_dataset(
            args.full_data_file,
            tokenizer,
            max_length=args.max_length,
            debug=args.debug
        )
        #seed = args.seed
        train_dataset, eval_dataset = full_dataset.train_test_split(test_size=0.1, seed=args.seed)
        if torch.cuda.current_device() == 0:
            print(f"Loaded {len(train_dataset)} examples for training")
            print(f"Loaded {len(eval_dataset)} examples for evaluation")
    print("\n3. Loading dataset done")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=None,  # We already padded to max_length
    )

    eval_steps = args.eval_steps if not args.pretrain else args.eval_steps * 20
    save_steps = len(train_dataset) // (world_size * args.batch_size * args.grad_accum)
    if args.same_flop:
        if args.jepa_ratio > 0.0:
            save_steps = int(save_steps / (1 + args.jepa_ratio))
            args.num_epochs = int(math.ceil(args.num_epochs / (1 + args.jepa_ratio)))
        elif args.additive_mask:
            save_steps = save_steps // 2
            args.num_epochs = int(math.ceil(args.num_epochs / 2))
        elif not args.regular:
            save_steps = save_steps // 3
            args.num_epochs = int(math.ceil(args.num_epochs / 3))
        if torch.cuda.current_device() == 0:
            print(f">>>>> --same_flop is active: Save checkpoint every: {save_steps} steps, run {args.num_epochs} epochs")
    output_dir = os.path.abspath(args.output_dir)
    if torch.cuda.current_device() == 0:
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training parameters
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        
        # Evaluation
        eval_strategy="no",  # "steps" if eval_dataset else "no",
        # eval_steps=eval_steps,
        
        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.num_epochs * 4,

        # Logging
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.eval_steps,
        
        # Optimization - key changes for stability
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,  # Enable for memory efficiency
        dataloader_drop_last=True,   # Drop last incomplete batch
        
        # Memory optimization
        dataloader_num_workers=0,    # Avoid multiprocessing issues
        
        # Multi-GPU settings - completely disable FSDP
        ddp_find_unused_parameters=False,
        ddp_backend="nccl" if world_size > 1 else None,
        
        # Explicitly disable FSDP and sharding
        fsdp="",
        fsdp_config={},
        
        # Other
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True if eval_dataset else False,
        
        # Disable problematic optimizations
        tf32=False,  # May help with stability
        
        # Set seed for reproducibility
        seed=args.seed,
        data_seed=args.seed,
    )
    if args.regular:
        if torch.cuda.current_device() == 0:
            print("\n3. Initializing regular trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # callbacks=[flop_callback] if args.track_flop else [],
        )
    else:
        trainer = JepaTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            lbd=args.lbd,
            gamma=args.gamma,
            last_token=args.last_token,
            debug=args.debug,
            additive_mask=args.additive_mask,
            jepa_l2=args.jepa_l2,
            jepa_mse=args.jepa_mse,
            infonce=args.infonce,
            jepa_ratio=args.jepa_ratio,
            step_jepa_predictors=args.predictors,
        )

        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)

        print(f"Trainable parameters: {len(trainable_params)}")
        if len(trainable_params) == 0:
            print("ERROR: No parameters require gradients!")
        else:
            print("First few trainable params:", trainable_params[:5])

    # Start training
    if torch.cuda.current_device() == 0:
        print("\n4. Starting training...")
    try:
        trainer.train()
    except Exception as e:
        if torch.cuda.current_device() == 0:
            print(f"Training failed with error: {e}")
            print("This might be due to FSDP/sharding issues. Try running with --lora flag for LoRA fine-tuning.")
        raise
    
    # Save final model
    if torch.cuda.current_device() == 0:
        print("\n5. Saving final model...")
if __name__ == "__main__":
    main()