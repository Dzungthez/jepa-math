# Debug: Loss Comparison between Regular and JEPA (lbd=0)

## Issue
User reports:
- `--regular`: loss 1.02 → 0.8
- `lbd=0`: llm_loss 1.0 → 0.5 (fast), then plateau

## Hypothesis
Possible causes:
1. Different effective batch sizes
2. Different loss normalization
3. Gradient variance from triple batch
4. Batch statistics in model layers

## Debug Steps

### Step 1: Check actual batch sizes

Add temporary debug at line 677 in `compute_loss()`:

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # Get all forward pass results
    forward_results = self.forward(model, inputs)

    # Extract main language modeling loss
    main_outputs = forward_results['main_outputs']
    lm_loss = main_outputs.loss

    # DEBUG: Print batch info
    if self.debug >= 5 and torch.cuda.current_device() == 0:
        actual_batch_size = inputs["input_ids"].shape[0]
        forward_batch_size = main_outputs.logits.shape[0]
        num_valid_labels = (main_outputs.logits != -100).sum().item()
        print(f"[DEBUG Loss] input_batch={actual_batch_size}, forward_batch={forward_batch_size}")
        print(f"  lm_loss={lm_loss.item():.4f}")
```

Run both:
```bash
# Regular
python train.py --train_file data.jsonl --regular --batch_size 4 --max_train_steps 10 --debug 5

# JEPA lbd=0
python train.py --train_file data.jsonl --lbd 0 --num_prediction_steps 2 --batch_size 4 --max_train_steps 10 --debug 5
```

Compare:
- Are batch sizes same?
- Are loss values consistently different?

### Step 2: Check loss per token

Model.loss is averaged over tokens. Check if number of tokens is same:

```python
# In compute_loss, after getting lm_loss
if self.debug >= 5:
    # Count valid tokens
    valid_tokens = (main_outputs.labels != -100).sum().item()
    print(f"  valid_tokens={valid_tokens}, loss_per_token={lm_loss.item()}")
```

### Step 3: Force same batch structure

Test if the issue is batch construction:

Modify `_build_triple_batch()` temporarily to use ONLY batch 1:
```python
# Instead of tripling, just use batch 1
return {
    "input_ids": batch1_input_ids,
    "labels": batch1_labels,
    "attention_mask": normal_causal_mask,  # Not tripled
}, True  # skip_jepa = True
```

This should behave exactly like regular mode. If loss is still different → problem elsewhere.

### Step 4: Check gradient accumulation

With grad_accum=4:
- Regular: 4 micro-batches of 4 samples = 16 samples per update
- JEPA lbd=0: 4 micro-batches of 4×3=12 samples forward, but only 4×4=16 contribute loss

Should be same effective batch size. But check optimizer.step() is called at same frequency.

## Expected Results

### If effective batch size is the issue:
- Regular: consistent gradient from all samples
- JEPA lbd=0: variance from only 1/3 of forward pass

### If normalization is the issue:
- Loss values will differ by constant factor
- E.g., JEPA loss = Regular loss / 3 (or × 3)

### If batch statistics is the issue:
- Different running means/vars in BatchNorm layers
- Different behavior in LayerNorm

## Recommendation

Most likely cause: **Effective batch size and gradient variance**.

**Solution:**
1. Use `--regular` for true SFT
2. If using JEPA, use `lbd > 0` to actually leverage the architecture
3. Don't use `lbd=0` as it wastes compute/memory without benefit

**Alternative:**
If you want to test with `lbd=0`, increase batch_size to compensate for variance:
```bash
# Instead of batch_size=4
python train.py --lbd 0 --batch_size 8  # or 12
```
