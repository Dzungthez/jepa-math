# Bug Fix: Fallback creates too many boundaries

## Problem

In `_build_triple_batch()` (line 428-439), the fallback logic creates:
```python
boundaries = [last_token * (j+1) // (num_parts + 1)
             for j in range(self.num_prediction_steps + 1)]
```

This ALWAYS creates `num_prediction_steps + 1` boundaries, even when the sample has fewer actual steps.

**Example:**
- User sets `--num_prediction_steps 100`
- Sample has only 8 "\n\n" separators (9 steps)
- `_find_step_boundaries()` returns None (can't find 101 boundaries)
- Fallback creates 101 artificial boundaries → 100 pairs per sample!
- Batch of 4 samples → 400 pairs total ❌

## Root Cause

The fallback doesn't consider the **actual number of steps in the sample**.

We should create `min(actual_steps, num_prediction_steps)` pairs, not always `num_prediction_steps`.

## Fix

Replace lines 428-439 in `jepa_trainer.py`:

```python
# BEFORE (BUGGY):
if boundaries is None or len(boundaries) < 2:
    # Fallback: create boundaries at equal intervals
    last_token = self._last_token_index(
        inputs["input_ids"][i:i+1],
        inputs["labels"][i:i+1],
        inputs["attention_mask"][i:i+1]
    )[0].item()

    # Divide sequence into num_prediction_steps + 1 parts
    num_parts = self.num_prediction_steps + 1
    boundaries = [last_token * (j+1) // (num_parts + 1)
                 for j in range(self.num_prediction_steps + 1)]

# AFTER (FIXED):
if boundaries is None or len(boundaries) < 2:
    # Fallback: create boundaries at equal intervals
    # But respect the actual number of steps in the sample

    # First, try to count actual "\n\n" separators in the sample
    ids = inputs["input_ids"][i].tolist()
    sep_tokens = self.processing_class.encode("\n\n", add_special_tokens=False)

    # Count all "\n\n" in sequence (not limited by num_steps)
    actual_separators = 0
    for j in range(len(ids) - len(sep_tokens) + 1):
        if ids[j:j+len(sep_tokens)] == sep_tokens:
            actual_separators += 1

    # Actual steps = separators + 1 (if we have separators)
    # But we can only form (actual_separators) pairs from (actual_separators+1) steps
    # e.g., 8 separators → 9 steps → 8 pairs max
    max_possible_pairs = actual_separators if actual_separators > 0 else 1

    # Create only as many boundaries as we need
    num_boundaries_to_create = min(self.num_prediction_steps + 1, max_possible_pairs + 1)

    last_token = self._last_token_index(
        inputs["input_ids"][i:i+1],
        inputs["labels"][i:i+1],
        inputs["attention_mask"][i:i+1]
    )[0].item()

    # Divide sequence into num_boundaries_to_create parts
    boundaries = [last_token * (j+1) // (num_boundaries_to_create + 1)
                 for j in range(num_boundaries_to_create)]
```

## Verification

After fix, with `--num_prediction_steps 100` and sample with 8 separators:
- `actual_separators = 8`
- `max_possible_pairs = 8`
- `num_boundaries_to_create = min(101, 9) = 9`
- `boundaries` has 9 elements → 8 pairs ✅
- Batch of 4 samples → ~32 pairs total ✅ (instead of 400)

## Alternative Simpler Fix

If you don't want to count separators again, you can use a simpler approach:

```python
if boundaries is None or len(boundaries) < 2:
    # Fallback: When we can't find enough step boundaries,
    # we should still limit pairs to a reasonable number

    # Estimate: most samples have 5-10 steps based on dataset stats
    # Use a conservative cap
    MAX_FALLBACK_PAIRS = 10  # Or use mean from dataset: 8

    num_boundaries_to_create = min(self.num_prediction_steps + 1, MAX_FALLBACK_PAIRS + 1)

    last_token = self._last_token_index(...)

    boundaries = [last_token * (j+1) // (num_boundaries_to_create + 1)
                 for j in range(num_boundaries_to_create)]
```

This caps fallback at 10 pairs per sample regardless of `num_prediction_steps`.

## Recommendation

I recommend the **first fix** (count actual separators) because it's more accurate and respects the actual structure of each sample.

The second fix is simpler but uses a hard-coded cap.
