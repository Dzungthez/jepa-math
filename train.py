import sys
import os
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling, default_data_collator
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
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation steps")
    parser.add_argument("--max_train_steps", type=int, default=-1,
                    help="Total number of training steps to perform. If > 0, overrides num_epochs.")

    
    # Step-JEPA arguments
    parser.add_argument("--step_jepa", action="store_true", help="Enable Step-JEPA mode (isolate Step 2)")
    parser.add_argument("--regular", action="store_true", help="Use regular trainer (SFT) without JEPA")
    parser.add_argument("--predictors", type=int, default=4, help="Number of K predictor tokens after Step 1")
    parser.add_argument("--lbd", type=float, default=0.5, help="Lambda for JEPA loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma for LM loss")
    parser.add_argument("--last_token", type=int, default=-1, help="Index of last token for embedding extraction")
    parser.add_argument("--jepa_l2", action="store_true", help="Use L2 norm as JEPA loss")
    parser.add_argument("--jepa_mse", action="store_true", help="Use MSE as JEPA loss")
    parser.add_argument("--infonce", action="store_true", help="Use InfoNCE loss")
    parser.add_argument("--jepa_ratio", type=float, default=-1.0, help="Random JEPA loss dropout ratio")
    parser.add_argument("--additive_mask", type = bool, default=True, help="Use additive mask")
    parser.add_argument("--unmask_user", action="store_true",
                       help="Enable LM loss on user content (in addition to assistant)")
    parser.add_argument("--view_based_jepa", action="store_true",
                       help="Use view-based JEPA (user_end vs assistant_end) instead of step-based")
    parser.add_argument("--num_prediction_steps", type=int, default=1,
                       help="Number of consecutive step pairs to predict (default=1). "
                            "Use 1 for current behavior, >1 for multi-step JEPA. "
                            "Incompatible with --view_based_jepa.")
    parser.add_argument("--use_localized_masks", type = bool, default=True,
                       help="Use localized step masks for multi-step JEPA (default: True). "
                            "When disabled, uses normal causal masks (saves memory: 2x vs 3x).")
    parser.add_argument("--no_localized_masks", action="store_false", dest="use_localized_masks",
                       help="Disable localized step masks (use normal causal instead).")
    parser.set_defaults(use_localized_masks=True)
    parser.add_argument("--include_last_step_target", type=bool, default=True,
                       help="Include assistant_end as target for last step (default: True). "
                            "When False, last step has no target (backward compatibility).")
    parser.add_argument("--omit_last_step_target", type=bool, default=False, dest="include_last_step_target",
                       help="Omit last step target (old behavior).")
    parser.add_argument("--nn_tokens_path", type=str, default="nn_tokens.json",
                       help="Path to JSON file containing token IDs for '\\n\\n'")

    # Other
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--debug", type=int, default=5, help="Debug level")
    parser.add_argument("--pretrain", action="store_true", help="Pretraining mode")
    parser.add_argument("--same_flop", action="store_true", help="Adjust epochs/steps to match FLOPs")
    
    args = parser.parse_args()

    # Validate incompatible flags
    if args.view_based_jepa and args.num_prediction_steps > 1:
        raise ValueError(
            "Cannot use both --view_based_jepa and --num_prediction_steps > 1. "
            "Multi-step JEPA requires step boundaries, which view-based mode doesn't have."
        )

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
        print(f"Unmask user content: {args.unmask_user}")
        print(f"JEPA mode: {'View-based' if args.view_based_jepa else 'Step-based'}")
        print(f"Number of prediction steps: {args.num_prediction_steps}")
        if args.num_prediction_steps > 1:
            print(f"Use localized masks: {args.use_localized_masks}")
        print(f"Include last step target: {args.include_last_step_target}")
        print(f"Last token offset: {args.last_token}")
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
            debug=args.debug,
            unmask_user=args.unmask_user
        )
        if args.eval_file is not None:
            eval_dataset = load_and_prepare_dataset(
                args.eval_file,
                tokenizer,
                max_length=args.max_length,
                debug=args.debug,
                unmask_user=args.unmask_user
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
            debug=args.debug,
            unmask_user=args.unmask_user
        )
        #seed = args.seed
        train_dataset, eval_dataset = full_dataset.train_test_split(test_size=0.1, seed=args.seed)
        if torch.cuda.current_device() == 0:
            print(f"Loaded {len(train_dataset)} examples for training")
            print(f"Loaded {len(eval_dataset)} examples for evaluation")
    print("\n3. Loading dataset done")

    steps_per_epoch = len(train_dataset) // (world_size * args.batch_size * args.grad_accum)
    if steps_per_epoch == 0:
        steps_per_epoch = 1

    use_max_steps = args.max_train_steps is not None and args.max_train_steps > 0
    if use_max_steps and torch.cuda.current_device() == 0:
        print(f">>>>> Using max_train_steps={args.max_train_steps} (overrides num_epochs={args.num_epochs})")
        print(f">>>>> steps_per_epoch = {steps_per_epoch}")

    if args.regular:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
    else:
        data_collator = default_data_collator

    eval_steps = args.eval_steps if not args.pretrain else args.eval_steps * 20
    save_steps = steps_per_epoch
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
    
    # Ensure save_steps is a multiple of eval_steps when load_best_model_at_end is enabled
    if eval_dataset:
        # Make save_steps a multiple of eval_steps
        save_steps = max(eval_steps, (save_steps // eval_steps) * eval_steps)
        if torch.cuda.current_device() == 0:
            print(f">>>>> Adjusted save_steps to {save_steps} (multiple of eval_steps={eval_steps})")
    
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
        num_train_epochs=1 if use_max_steps else args.num_epochs,
        max_steps=args.max_train_steps if use_max_steps else -1,
        
        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
        
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
        remove_unused_columns=True if args.regular else False,
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
            unmask_user=args.unmask_user,
            view_based_jepa=args.view_based_jepa,
            num_prediction_steps=args.num_prediction_steps,
            use_localized_masks=args.use_localized_masks,
            include_last_step_target=args.include_last_step_target,
            nn_tokens_path=args.nn_tokens_path,
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