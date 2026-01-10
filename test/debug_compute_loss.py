"""
Debug script to trace the exact values in compute_loss() during triple batch.
Add this snippet to jepa_trainer.py temporarily to debug.

Put this at line 739 (inside compute_loss, triple batch section):

    # ===== DEBUG: Print metadata =====
    if torch.cuda.current_device() == 0:
        print(f"\n[DEBUG compute_loss] Triple batch processing:")
        print(f"  total_batch_size (jepa_hidden_states.shape[0]): {total_batch_size}")
        print(f"  computed batch_size: {batch_size}")
        print(f"  len(self._num_pairs_per_sample): {len(self._num_pairs_per_sample)}")
        print(f"  self._num_pairs_per_sample: {self._num_pairs_per_sample}")
        print(f"  sum of pairs: {sum(self._num_pairs_per_sample)}")
    # ===== END DEBUG =====

This will show:
- Actual batch_size being used
- The _num_pairs_per_sample list
- How many pairs will be processed

If you see:
- batch_size = 200 → BUG! Should be 4
- _num_pairs_per_sample has 200 items → BUG! Should have 4
- sum(_num_pairs_per_sample) = 400 → This is where the 400 comes from

INSTRUCTIONS:
1. Add the debug snippet to jepa_trainer.py at line 739
2. Run training with --debug 5 --max_train_steps 1
3. Check the output
4. Report back the values
"""

print(__doc__)
