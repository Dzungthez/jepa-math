from transformers import Trainer
import torch 
import torch.nn.functional as F
class JepaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Extract custom loss parameters
        self.lbd = kwargs.pop('lbd', 1.0)
        self.gamma = kwargs.pop('gamma', 1.0)
        self.last_token = kwargs.pop('last_token', -2)
        self.debug = kwargs.pop('debug', 0)
        self.additive_mask = kwargs.pop('additive_mask', False)
        self.jepa_l2 = kwargs.pop('jepa_l2', False)
        self.jepa_mse = kwargs.pop('jepa_mse', False)
        self.infonce = kwargs.pop('infonce', False)
        self.jepa_ratio = kwargs.pop('jepa_ratio', -1.0)
        self.step_jepa_predictors = kwargs.pop('step_jepa_predictors', 3)
        self.unmask_user = kwargs.pop('unmask_user', False)
        self.view_based_jepa = kwargs.pop('view_based_jepa', False)
        self.num_prediction_steps = kwargs.pop('num_prediction_steps', 1)
        self.use_localized_masks = kwargs.pop('use_localized_masks', True)
        self.include_last_step_target = kwargs.pop('include_last_step_target', True)

        # Load "\n\n" token IDs (collected offline)
        nn_tokens_path = kwargs.pop('nn_tokens_path', 'nn_tokens.json')
        self.nn_token_ids = self._load_nn_tokens(nn_tokens_path)

        assert self.jepa_l2 + self.jepa_mse <= 1, "Only one of jepa_l2 and jepa_mse can be True."

        # Validation: view_based_jepa incompatible with multi-step
        if self.view_based_jepa and self.num_prediction_steps > 1:
            raise ValueError(
                "Cannot use both --view_based_jepa and --num_prediction_steps > 1. "
                "Multi-step JEPA requires step boundaries from '\\n\\n' separators."
            )
        # Debugging
        self.print_nn_positions = kwargs.pop("print_nn_positions", False)
        self.print_nn_max_steps = kwargs.pop("print_nn_max_steps", 5)   # chỉ in 5 bước đầu
        self.print_nn_max_per_sample = kwargs.pop("print_nn_max_per_sample", 10)
        self._printed_nn_steps = 0

        self.print_last_step_target_debug = kwargs.pop("print_last_step_target_debug", False)
        self.print_last_step_target_max = kwargs.pop("print_last_step_target_max", 10)
        self._print_last_step_target_count = 0


        super().__init__(*args, **kwargs)

    def _debug_print_nn(self, tokenizer, input_ids_1d, assistant_start, assistant_end, sample_idx=0):
        """
        In ra các vị trí token nằm trong nn_token_ids, và decode cửa sổ quanh nó để verify.
        """
        if not self.print_nn_positions:
            return

        # chỉ in ở gpu0 và giới hạn số step
        if torch.cuda.is_available():
            if torch.cuda.current_device() != 0:
                return
        if self._printed_nn_steps >= self.print_nn_max_steps:
            return

        ids = input_ids_1d.tolist()
        hits = []
        for pos in range(assistant_start, assistant_end + 1):
            if ids[pos] in self.nn_token_ids:
                hits.append(pos)
                if len(hits) >= self.print_nn_max_per_sample:
                    break

        print(f"\n[NN-DEBUG] sample={sample_idx} assistant_span=[{assistant_start},{assistant_end}] "
            f"hits={len(hits)} positions={hits}")

        # decode từng hit với context window
        window = 8  # số token trước/sau để nhìn
        for pos in hits:
            lo = max(assistant_start, pos - window)
            hi = min(assistant_end, pos + window)
            chunk_ids = ids[lo:hi+1]

            # decode cửa sổ
            chunk_txt = tokenizer.decode(chunk_ids, skip_special_tokens=False)
            tok_txt = tokenizer.decode([ids[pos]], skip_special_tokens=False)

            # show rõ vị trí token trong cửa sổ bằng cách decode prefix/suffix
            prefix = tokenizer.decode(ids[lo:pos], skip_special_tokens=False)
            suffix = tokenizer.decode(ids[pos+1:hi+1], skip_special_tokens=False)

            print(f"  - pos={pos} token_id={ids[pos]}")
            print(f"    token_decoded: {repr(tok_txt)}")
            print(f"    window[{lo}:{hi}]: {repr(chunk_txt)}")
            print(f"    split: {repr(prefix)} || {repr(tok_txt)} || {repr(suffix)}")

        self._printed_nn_steps += 1


    def _load_nn_tokens(self, path):
        """Load pre-collected token IDs that contain '\n\n'"""
        import json
        import os

        if not os.path.exists(path):
            print(f"WARNING: nn_tokens.json not found at {path}, using fallback")
            # Fallback: common tokens for Llama-3.2
            return [1473, 2595, 45464, 35432, 2266, 43115, 95181, 271, 57277, 7887, 9456, 1363, 2195, 696, 15804, 1980, 3677, 382]

        with open(path, 'r') as f:
            data = json.load(f)

        nn_token_ids = set(data['nn_token_ids'])
        print(f"Loaded {len(nn_token_ids)} '\n\n' token IDs from {path}")
        print(f"Token IDs: {sorted(nn_token_ids)}")

        return nn_token_ids

    def _build_additive_mask(self, k: int):
        mask = torch.zeros((k, k), dtype=torch.float32)
        mask[torch.triu(torch.ones(k, k), diagonal=1) == 1] = -torch.inf
        return mask
    
    def _last_token_index(self, input_ids, labels, attention_mask):
        """Find the index of the last non-padding token in each sequence."""
        index = []
        def unpad(input_ids, attention_mask):
            result = []
            can_break = False
            for id, mask in zip(input_ids, attention_mask):
                if mask != 0:
                    can_break = True
                if mask == 0 and can_break:
                    break
                result.append(id)
            return result

        for i in range(input_ids.shape[0]):
            uii = unpad(input_ids[i], attention_mask[i])
            index.append(len(uii) + self.last_token)
        
        index_tensor = torch.tensor(index).to(input_ids.device)
        return index_tensor
    
    def _find_step_boundaries(self, input_ids, labels, tokenizer, user_span=None,
                              assistant_span=None, num_steps=2):
        """
        Find step boundaries for step-based JEPA.
        Uses metadata if available, falls back to label-based detection.

        CRITICAL: Always searches for "\\n\\n" ONLY within assistant span.
        """
        ids = input_ids.tolist()
        lab = labels.tolist()

        # Determine assistant span
        if assistant_span is not None:
            assistant_start, assistant_end = assistant_span
        else:
            # Fallback: detect assistant boundary from chat template markers
            # This is robust even when unmask_user=True (doesn't rely on labels)

            # Debug warning when fallback is triggered
            if self.debug >= 3 and torch.cuda.current_device() == 0:
                print(f"WARNING: Using fallback boundary detection (metadata missing)")
                if self.unmask_user:
                    print(f"         unmask_user=True detected, using chat template markers")

            # 1. Find assistant header in input_ids
            assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            assistant_header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)

            assistant_start = None
            for i in range(len(ids) - len(assistant_header_ids) + 1):
                if ids[i:i+len(assistant_header_ids)] == assistant_header_ids:
                    # Position after header = assistant content start
                    assistant_start = i + len(assistant_header_ids)
                    break

            if assistant_start is None:
                # Could not find assistant header, unsafe to proceed
                if self.debug >= 3 and torch.cuda.current_device() == 0:
                    print(f"WARNING: Could not find assistant header in fallback, returning None")
                return None

            # 2. Find assistant end (last non-padding token)
            assistant_end = None
            for i in range(len(ids) - 1, -1, -1):
                # Check if this is a real token (not padding)
                if ids[i] != tokenizer.pad_token_id:
                    assistant_end = i
                    break

            if assistant_end is None or assistant_end <= assistant_start:
                return None

        # Search for "\\n\\n" ONLY within assistant span
        # Use precomputed token set 
        occ = []

        for i in range(assistant_start, assistant_end):
            # Simple set membership check (very fast!)
            if ids[i] in self.nn_token_ids:
                occ.append(i)  # Found a token containing "\n\n"

                # Only need num_steps - 1 separators
                if len(occ) >= num_steps - 1:
                    break
        if self.print_last_step_target_debug and (not torch.cuda.is_available() or torch.cuda.current_device() == 0):
            if self._print_last_step_target_count < self.print_last_step_target_max:
                self._print_last_step_target_count += 1
                print("[LAST_STEP_TARGET_DEBUG] occ(separators found) =", occ, 
                            " (assistant_end candidate =", assistant_end, ")")

        # Add assistant_end as final boundary (if enabled)
        if self.include_last_step_target:
            if len(occ) > 0:
                # Have separators, add assistant_end as final boundary
                occ.append(assistant_end)
            elif num_steps == 1:
                # Special case: single step with no separators
                occ = [assistant_end]

        if self.print_last_step_target_debug and (not torch.cuda.is_available() or torch.cuda.current_device() == 0):
            if self._print_last_step_target_count < self.print_last_step_target_max:
                print("[LAST_STEP_TARGET_DEBUG] occ(after maybe append assistant_end) =", occ)


        # Apply last_token offset to ALL boundaries
        adjusted_occ = []
        for boundary in occ:
            adjusted_boundary = boundary + self.last_token
            # Clamp to valid range [0, assistant_end]
            adjusted_boundary = max(0, min(adjusted_boundary, assistant_end))
            adjusted_occ.append(adjusted_boundary)

        if self.print_last_step_target_debug and (not torch.cuda.is_available() or torch.cuda.current_device() == 0):
            if self._print_last_step_target_count < self.print_last_step_target_max:
                print("[LAST_STEP_TARGET_DEBUG] adjusted_occ =", adjusted_occ)
                if len(occ) > 0:
                    print("[LAST_STEP_TARGET_DEBUG] last_raw_boundary =", occ[-1], 
                        " last_adjusted_boundary =", adjusted_occ[-1])
                self._print_last_step_target_count += 1


        # Debug print positions of tokens that contain "\n\n"
        if self.print_nn_positions:
            self._debug_print_nn(tokenizer, input_ids, assistant_start, assistant_end, sample_idx=-1)

        # Need at least num_steps boundaries to form prediction pairs
        if len(adjusted_occ) < num_steps:
            return None

        return adjusted_occ  # Return adjusted list

    def _find_view_boundaries(self, input_ids, labels, user_span=None, assistant_span=None):
        """
        Find view boundaries for view-based JEPA.

        View 1 = end of user content
        View 2 = end of assistant content
        """
        lab = labels.tolist()

        if user_span is not None and assistant_span is not None:
            user_start, user_end = user_span
            assistant_start, assistant_end = assistant_span

            if user_end < user_start or assistant_end < assistant_start:
                return None, None

            return user_end, assistant_end

        # Fallback: cannot reliably determine without metadata
        return None, None

    def _build_double_batch(self, inputs, user_spans, assistant_spans):
        """
        Current implementation: Double batch for single-step prediction.
        Used when num_prediction_steps == 1.

        Batch 1: LM loss (original sequences, normal causal mask)
        Batch 2: JEPA loss (sequences with predictors, isolated view2 mask)
        """
        batch_size = inputs["input_ids"].shape[0]
        device = inputs["input_ids"].device

        # Find boundaries for each example (mode-dependent)
        view1_end_positions = []  # step1_end or user_end
        view2_end_positions = []  # step2_end or assistant_end

        for i in range(batch_size):
            # Extract spans for this sample
            user_span = None
            assistant_span = None

            if user_spans is not None:
                user_span = (user_spans[i][0].item(), user_spans[i][1].item())
            if assistant_spans is not None:
                assistant_span = (assistant_spans[i][0].item(), assistant_spans[i][1].item())

            if self.view_based_jepa:
                # View-based mode: view1 = user_end, view2 = assistant_end
                view1_end, view2_end = self._find_view_boundaries(
                    inputs["input_ids"][i],
                    inputs["labels"][i],
                    user_span=user_span,
                    assistant_span=assistant_span
                )

                if view1_end is None or view2_end is None:
                    # Fallback strategy for view-based
                    print(f"Fallback to sequence thirds for view-based mode")
                    last_token = self._last_token_index(
                        inputs["input_ids"][i:i+1],
                        inputs["labels"][i:i+1],
                        inputs["attention_mask"][i:i+1]
                    )[0].item()

                    view1_end = last_token // 2
                    view2_end = last_token
            else:
                # Step-based mode: view1 = step1_end, view2 = step2_end
                boundaries = self._find_step_boundaries(
                    inputs["input_ids"][i],
                    inputs["labels"][i],
                    self.processing_class,
                    user_span=user_span,
                    assistant_span=assistant_span,
                    num_steps=2  # For double batch, only need first 2 boundaries
                )

                if boundaries is not None:
                    view1_end = boundaries[0]
                    view2_end = boundaries[1]
                else:
                    # Fallback to sequence division
                    last_token = self._last_token_index(
                        inputs["input_ids"][i:i+1],
                        inputs["labels"][i:i+1],
                        inputs["attention_mask"][i:i+1]
                    )[0].item()

                    if self.include_last_step_target:
                        # New behavior: include last_token position as final boundary
                        view1_end = last_token // 2
                        view2_end = last_token
                    else:
                        # Old behavior: omit final boundary
                        view1_end = last_token // 3
                        view2_end = (last_token * 2) // 3

            view1_end_positions.append(view1_end)
            view2_end_positions.append(view2_end)

        view1_end_pos = torch.tensor(view1_end_positions, device=device)
        view2_end_pos = torch.tensor(view2_end_positions, device=device)

        # Insert K predictor tokens after view1 for each example
        # Create new input_ids with predictor tokens inserted (for JEPA batch only)
        new_input_ids_with_pred = []
        for i in range(batch_size):
            view1_end = view1_end_pos[i].item()
            # Insert K predictor tokens after view1_end
            predictor_ids = [self.processing_class.convert_tokens_to_ids(f"<|predictor_{j+1}|>")
                            for j in range(self.step_jepa_predictors)]

            new_seq = torch.cat([
                inputs["input_ids"][i, :view1_end+1],
                torch.tensor(predictor_ids, device=device),
                inputs["input_ids"][i, view1_end+1:]
            ])
            # Most sequences have padding at the end
            new_input_ids_with_pred.append(new_seq)

        # Stack with padding to handle variable lengths (for sequences with predictors)
        max_len = max(seq.shape[0] for seq in new_input_ids_with_pred)
        padded_input_ids_with_pred = []
        for seq in new_input_ids_with_pred:
            if seq.shape[0] < max_len:
                padding = torch.full((max_len - seq.shape[0],),
                                    self.processing_class.pad_token_id,
                                    device=device)
                seq = torch.cat([seq, padding])
            padded_input_ids_with_pred.append(seq)

        new_input_ids_with_pred = torch.stack(padded_input_ids_with_pred)

        # Pad original input_ids to match the new length (if needed)
        original_input_ids = inputs["input_ids"]
        if original_input_ids.shape[1] < max_len:
            padding = torch.full((batch_size, max_len - original_input_ids.shape[1]),
                                self.processing_class.pad_token_id,
                                device=device)
            original_input_ids = torch.cat([original_input_ids, padding], dim=1)

        # DOUBLE THE BATCH
        # First half: Original sequences WITHOUT predictor tokens (for LM loss)
        # Second half: Modified sequences WITH predictor tokens (for JEPA loss)
        doubled_input_ids = torch.cat([original_input_ids, new_input_ids_with_pred], dim=0)

        # Pad labels to match new length if needed
        original_labels = inputs["labels"]
        if original_labels.shape[1] < max_len:
            padding = torch.full((batch_size, max_len - original_labels.shape[1]),
                                -100,  # Mask padded positions
                                device=device)
            original_labels = torch.cat([original_labels, padding], dim=1)

        # Batch 2 (JEPA) doesn't need LM loss, so mask all labels
        jepa_labels = torch.full_like(original_labels, -100)

        doubled_labels = torch.cat([original_labels, jepa_labels], dim=0)  # Only batch 1 computes LM loss

        # Create attention masks for DOUBLED batch
        # Use max_len instead of seq_length since sequences may be longer now
        mask = torch.full((batch_size * 2, 1, max_len, max_len), float('-inf')).to(device)

        for i in range(batch_size):
            view1_end = view1_end_pos[i].item()
            view2_end = view2_end_pos[i].item()

            # Find actual sequence length (non-padding) from original input
            last_token = self._last_token_index(
                inputs["input_ids"][i:i+1],
                inputs["labels"][i:i+1],
                inputs["attention_mask"][i:i+1]
            )[0].item()

            # FIRST HALF (index i): Original sequences WITHOUT predictors
            # Normal causal mask for entire original sequence
            seq_len_original = last_token + 1
            mask[i, 0, :seq_len_original, :seq_len_original] = self._build_additive_mask(seq_len_original)

            # SECOND HALF (index i + batch_size): Modified sequences WITH predictors
            # JEPA isolation mask (step-based or view-based)

            # Calculate positions after inserting K predictor tokens
            predictor_start = view1_end + 1
            predictor_end = predictor_start + self.step_jepa_predictors - 1
            view2_start = predictor_end + 1
            view2_end_adjusted = view2_end + self.step_jepa_predictors
            seq_len_with_pred = last_token + 1 + self.step_jepa_predictors

            # - Everything before view2: normal causal
            mask[i + batch_size, 0, :view2_start, :view2_start] = self._build_additive_mask(view2_start)
            # - View2: isolated (can only see itself)
            mask[i + batch_size, 0, view2_start:view2_end_adjusted+1, view2_start:view2_end_adjusted+1] = \
                self._build_additive_mask(view2_end_adjusted - view2_start + 1)
            # - After view2: normal causal (can see everything)
            if view2_end_adjusted + 1 < seq_len_with_pred:
                mask[i + batch_size, 0, view2_end_adjusted+1:seq_len_with_pred, :seq_len_with_pred] = \
                    self._build_additive_mask(seq_len_with_pred)[view2_end_adjusted+1:seq_len_with_pred, :seq_len_with_pred]

        # Store positions for later use in compute_loss
        # These are for the SECOND HALF of the doubled batch
        self._view1_end_pos = view1_end_pos
        self._view2_end_pos = view2_end_pos + self.step_jepa_predictors  # Adjusted for inserted tokens
        self._predictor_pos = view1_end_pos + self.step_jepa_predictors  # Last predictor token

        return {
            "input_ids": doubled_input_ids,      # Shape: (batch_size * 2, seq_len)
            "labels": doubled_labels,            # Shape: (batch_size * 2, seq_len)
            "attention_mask": mask,              # Shape: (batch_size * 2, 1, seq_len, seq_len)
        }, False

    def _build_double_batch_no_isolation(self, inputs, user_spans, assistant_spans):
        """
        Double batch for single-step prediction WITHOUT view2 isolation.

        Batch 1: LM loss (original sequences, normal causal)
        Batch 2: Predictors inserted (normal causal for entire sequence, including view2)

        This is simpler than isolated view2 - view2 can see full causal context.
        """
        batch_size = inputs["input_ids"].shape[0]
        device = inputs["input_ids"].device

        # Find view boundaries
        view1_end_pos = []
        view2_end_pos = []

        for i in range(batch_size):
            user_span = None
            assistant_span = None
            if user_spans is not None:
                user_span = (user_spans[i][0].item(), user_spans[i][1].item())
            if assistant_spans is not None:
                assistant_span = (assistant_spans[i][0].item(), assistant_spans[i][1].item())

            # Find view boundaries (reuse existing logic)
            if self.view_based_jepa:
                view1_end, view2_end = self._find_view_boundaries(
                    inputs["input_ids"][i],
                    inputs["labels"][i],
                    user_span=user_span,
                    assistant_span=assistant_span
                )

                if view1_end is None or view2_end is None:
                    # Fallback strategy for view-based
                    last_token = self._last_token_index(
                        inputs["input_ids"][i:i+1],
                        inputs["labels"][i:i+1],
                        inputs["attention_mask"][i:i+1]
                    )[0].item()
                    view1_end = last_token // 2
                    view2_end = last_token
            else:
                # Step-based mode
                boundaries = self._find_step_boundaries(
                    inputs["input_ids"][i],
                    inputs["labels"][i],
                    self.processing_class,
                    user_span=user_span,
                    assistant_span=assistant_span,
                    num_steps=2
                )

                if boundaries is not None:
                    view1_end = boundaries[0]
                    view2_end = boundaries[1]
                else:
                    # Fallback to sequence division
                    last_token = self._last_token_index(
                        inputs["input_ids"][i:i+1],
                        inputs["labels"][i:i+1],
                        inputs["attention_mask"][i:i+1]
                    )[0].item()

                    if self.include_last_step_target:
                        # New behavior: include last_token position as final boundary
                        view1_end = last_token // 2
                        view2_end = last_token
                    else:
                        # Old behavior: omit final boundary
                        view1_end = last_token // 3
                        view2_end = (last_token * 2) // 3

            view1_end_pos.append(view1_end)
            view2_end_pos.append(view2_end)

        # === BATCH 1: LM Loss ===
        batch1_input_ids = inputs["input_ids"]
        batch1_labels = inputs["labels"]

        # === BATCH 2: Insert predictors ===
        batch2_input_ids_list = []

        for i in range(batch_size):
            view1_end = view1_end_pos[i]
            original_seq = inputs["input_ids"][i]

            # Insert K predictor tokens after view1
            predictor_ids = [
                self.processing_class.convert_tokens_to_ids(f"<|predictor_{j+1}|>")
                for j in range(self.step_jepa_predictors)
            ]
            predictor_tensor = torch.tensor(predictor_ids, device=device)

            # Split and insert
            new_seq = torch.cat([
                original_seq[:view1_end+1],
                predictor_tensor,
                original_seq[view1_end+1:]
            ])

            batch2_input_ids_list.append(new_seq)

        # Pad batch2 sequences
        max_len_batch2 = max(seq.shape[0] for seq in batch2_input_ids_list)
        batch2_input_ids = torch.full(
            (batch_size, max_len_batch2),
            self.processing_class.pad_token_id,
            device=device
        )
        for i, seq in enumerate(batch2_input_ids_list):
            batch2_input_ids[i, :seq.shape[0]] = seq

        # Batch2 labels: all masked
        batch2_labels = torch.full((batch_size, max_len_batch2), -100, device=device)

        # === Pad both batches to same max_length ===
        max_len = max(batch1_input_ids.shape[1], batch2_input_ids.shape[1])

        def pad_to_length(tensor, target_len, pad_value):
            if tensor.shape[1] < target_len:
                padding = torch.full(
                    (tensor.shape[0], target_len - tensor.shape[1]),
                    pad_value,
                    device=device
                )
                return torch.cat([tensor, padding], dim=1)
            return tensor

        batch1_input_ids = pad_to_length(batch1_input_ids, max_len, self.processing_class.pad_token_id)
        batch1_labels = pad_to_length(batch1_labels, max_len, -100)
        batch2_input_ids = pad_to_length(batch2_input_ids, max_len, self.processing_class.pad_token_id)
        batch2_labels = pad_to_length(batch2_labels, max_len, -100)

        # === Create attention masks (NORMAL CAUSAL for entire batch 2) ===
        doubled_mask = torch.full((batch_size * 2, 1, max_len, max_len), float('-inf')).to(device)

        for i in range(batch_size):
            last_token = self._last_token_index(
                inputs["input_ids"][i:i+1],
                inputs["labels"][i:i+1],
                inputs["attention_mask"][i:i+1]
            )[0].item()

            # BATCH 1: Normal causal
            seq_len = last_token + 1
            doubled_mask[i, 0, :seq_len, :seq_len] = self._build_additive_mask(seq_len)

            # BATCH 2: Normal causal (with predictors) - NO ISOLATION
            seq_len_with_pred = seq_len + self.step_jepa_predictors
            doubled_mask[i + batch_size, 0, :seq_len_with_pred, :seq_len_with_pred] = \
                self._build_additive_mask(seq_len_with_pred)

        # Store metadata for loss computation
        self._view1_end_pos = torch.tensor(view1_end_pos, device=device)
        self._view2_end_pos = torch.tensor(view2_end_pos, device=device) + self.step_jepa_predictors
        self._predictor_pos = torch.tensor(view1_end_pos, device=device) + self.step_jepa_predictors

        # Double batch
        doubled_input_ids = torch.cat([batch1_input_ids, batch2_input_ids], dim=0)
        doubled_labels = torch.cat([batch1_labels, batch2_labels], dim=0)

        return {
            "input_ids": doubled_input_ids,
            "labels": doubled_labels,
            "attention_mask": doubled_mask,
        }, False

    def _build_localized_step_masks(self, step_boundaries, seq_length):
        """
        Create attention mask where each step is isolated.

        Each step can only attend to tokens within the same step (causal within step).
        Tokens before the first step (system, user content) use normal causal attention.

        Args:
            step_boundaries: List [step0_end, step1_end, ..., stepN_end]
                            These are end positions of each step in assistant response
            seq_length: Total sequence length

        Returns:
            mask: (seq_length, seq_length) attention mask with shape compatible with model

        Example:
            If step_boundaries = [10, 20, 30] and seq_length = 40:
            - Positions 0-10: Step 0, can attend to 0-10 (causal)
            - Positions 11-20: Step 1, can attend to 11-20 (causal, isolated from step 0)
            - Positions 21-30: Step 2, can attend to 21-30 (causal, isolated from steps 0,1)
            - Positions 31-40: After last step, normal causal (can see everything)
        """
        mask = torch.full((seq_length, seq_length), float('-inf'))

        if len(step_boundaries) == 0:
            # No step boundaries found, use normal causal mask
            return self._build_additive_mask(seq_length)

        # Define step ranges
        # First step starts at position 0
        step_starts = [0] + [b + 1 for b in step_boundaries[:-1]]
        step_ends = step_boundaries

        # Apply causal mask within each step (steps are isolated from each other)
        for start, end in zip(step_starts, step_ends):
            step_len = end - start + 1
            causal_mask = self._build_additive_mask(step_len)
            mask[start:end+1, start:end+1] = causal_mask

        # Tokens after the last step boundary can see everything (normal causal)
        last_boundary = step_boundaries[-1]
        if last_boundary + 1 < seq_length:
            remaining_len = seq_length - (last_boundary + 1)
            causal_mask = self._build_additive_mask(seq_length)
            # Tokens after last_boundary use normal causal (can see all previous)
            mask[last_boundary+1:seq_length, :seq_length] = \
                causal_mask[last_boundary+1:seq_length, :seq_length]

        return mask

    def _build_triple_batch(self, inputs, user_spans, assistant_spans):
        """
        New implementation: Triple batch for multi-step prediction.

        Batch 1: LM loss (original sequences, normal causal mask)
        Batch 2: Predictor embeddings (insert predictors after each step, normal causal mask)
        Batch 3: Target embeddings (original sequences, localized step masks)

        For num_prediction_steps=N:
        - Find N+1 step boundaries (to form N consecutive pairs)
        - Insert K predictor tokens after each of first N steps
        - Batch 3 uses localized masks where each step is isolated
        """
        batch_size = inputs["input_ids"].shape[0]
        device = inputs["input_ids"].device

        # Find step boundaries for each sample
        all_step_boundaries = []  # List of lists
        num_pairs_per_sample = []  # Track how many prediction pairs each sample has

        for i in range(batch_size):
            # Extract spans for this sample
            user_span = None
            assistant_span = None

            if user_spans is not None:
                user_span = (user_spans[i][0].item(), user_spans[i][1].item())
            if assistant_spans is not None:
                assistant_span = (assistant_spans[i][0].item(), assistant_spans[i][1].item())

            # Find up to num_prediction_steps + 1 boundaries
            boundaries = self._find_step_boundaries(
                inputs["input_ids"][i],
                inputs["labels"][i],
                self.processing_class,
                user_span=user_span,
                assistant_span=assistant_span,
                num_steps=self.num_prediction_steps + 1  # Need N+1 boundaries for N pairs
            )

            if boundaries is None or len(boundaries) < 2:
                # Fallback: create boundaries at equal intervals
                # But respect the actual number of steps in the sample

                # First, count actual "\n\n" separators in the sample
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
                if self.include_last_step_target:
                    # New behavior: include last_token position as final boundary
                    boundaries = [last_token * (j+1) // (num_boundaries_to_create + 1)
                                 for j in range(num_boundaries_to_create - 1)]
                    boundaries.append(last_token)  # Add exact last_token position
                else:
                    # Old behavior: omit final boundary
                    boundaries = [last_token * (j+1) // (num_boundaries_to_create + 1)
                                 for j in range(num_boundaries_to_create)]

            # Actual number of pairs = len(boundaries) - 1
            num_pairs = len(boundaries) - 1

            all_step_boundaries.append(boundaries)
            num_pairs_per_sample.append(num_pairs)

        # ===== BATCH 1: LM Loss =====
        batch1_input_ids = inputs["input_ids"]
        batch1_labels = inputs["labels"]

        # ===== BATCH 2: Insert predictors after each step (except last) =====
        batch2_input_ids_list = []
        predictor_positions_list = []  # Track where predictors are inserted

        for i in range(batch_size):
            boundaries = all_step_boundaries[i]
            num_pairs = num_pairs_per_sample[i]

            # Insert K predictor tokens after each of first N steps
            original_seq = inputs["input_ids"][i]
            new_seq_parts = []
            predictor_positions = []
            cumulative_offset = 0

            prev_pos = 0
            for step_idx in range(num_pairs):  # Insert after steps 0, 1, ..., N-1
                step_end = boundaries[step_idx]

                # Add tokens up to step_end
                new_seq_parts.append(original_seq[prev_pos:step_end+1])

                # Insert K predictor tokens
                predictor_ids = [
                    self.processing_class.convert_tokens_to_ids(f"<|predictor_{j+1}|>")
                    for j in range(self.step_jepa_predictors)
                ]
                new_seq_parts.append(torch.tensor(predictor_ids, device=device))

                # Track last predictor position (adjusted for cumulative insertions)
                predictor_pos = step_end + cumulative_offset + self.step_jepa_predictors
                predictor_positions.append(predictor_pos)
                cumulative_offset += self.step_jepa_predictors

                prev_pos = step_end + 1

            # Add remaining tokens after last predictor
            new_seq_parts.append(original_seq[prev_pos:])
            new_seq = torch.cat(new_seq_parts)

            batch2_input_ids_list.append(new_seq)
            predictor_positions_list.append(predictor_positions)

        # Pad batch2 sequences to same length
        max_len_batch2 = max(seq.shape[0] for seq in batch2_input_ids_list)
        batch2_input_ids = torch.full(
            (batch_size, max_len_batch2),
            self.processing_class.pad_token_id,
            device=device
        )
        for i, seq in enumerate(batch2_input_ids_list):
            batch2_input_ids[i, :seq.shape[0]] = seq

        # Batch2 labels: all masked (no LM loss)
        batch2_labels = torch.full((batch_size, max_len_batch2), -100, device=device)

        # ===== BATCH 3: Localized step masks =====
        batch3_input_ids = inputs["input_ids"]
        batch3_labels = torch.full_like(inputs["labels"], -100)  # No LM loss

        # ===== Pad all batches to same max_length =====
        max_len = max(
            batch1_input_ids.shape[1],
            batch2_input_ids.shape[1],
            batch3_input_ids.shape[1]
        )

        def pad_to_length(tensor, target_len, pad_value):
            if tensor.shape[1] < target_len:
                padding = torch.full(
                    (tensor.shape[0], target_len - tensor.shape[1]),
                    pad_value,
                    device=device
                )
                return torch.cat([tensor, padding], dim=1)
            return tensor

        batch1_input_ids = pad_to_length(batch1_input_ids, max_len, self.processing_class.pad_token_id)
        batch1_labels = pad_to_length(batch1_labels, max_len, -100)

        batch2_input_ids = pad_to_length(batch2_input_ids, max_len, self.processing_class.pad_token_id)
        batch2_labels = pad_to_length(batch2_labels, max_len, -100)

        batch3_input_ids = pad_to_length(batch3_input_ids, max_len, self.processing_class.pad_token_id)
        batch3_labels = pad_to_length(batch3_labels, max_len, -100)

        # ===== Create attention masks =====
        tripled_mask = torch.full((batch_size * 3, 1, max_len, max_len), float('-inf')).to(device)

        for i in range(batch_size):
            # Find actual sequence length
            last_token = self._last_token_index(
                inputs["input_ids"][i:i+1],
                inputs["labels"][i:i+1],
                inputs["attention_mask"][i:i+1]
            )[0].item()
            seq_len = last_token + 1

            # BATCH 1: Normal causal mask
            tripled_mask[i, 0, :seq_len, :seq_len] = self._build_additive_mask(seq_len)

            # BATCH 2: Normal causal mask (with predictors)
            seq_len_with_pred = seq_len + num_pairs_per_sample[i] * self.step_jepa_predictors
            tripled_mask[i + batch_size, 0, :seq_len_with_pred, :seq_len_with_pred] = \
                self._build_additive_mask(seq_len_with_pred)

            # BATCH 3: Localized step masks
            boundaries = all_step_boundaries[i]
            tripled_mask[i + 2*batch_size, 0, :seq_len, :seq_len] = \
                self._build_localized_step_masks(boundaries, seq_len)

        # Store metadata for loss computation
        self._all_step_boundaries = all_step_boundaries
        self._predictor_positions = predictor_positions_list
        self._num_pairs_per_sample = num_pairs_per_sample

        # Triple batch
        tripled_input_ids = torch.cat([batch1_input_ids, batch2_input_ids, batch3_input_ids], dim=0)
        tripled_labels = torch.cat([batch1_labels, batch2_labels, batch3_labels], dim=0)

        return {
            "input_ids": tripled_input_ids,      # Shape: (batch_size * 3, seq_len)
            "labels": tripled_labels,            # Shape: (batch_size * 3, seq_len)
            "attention_mask": tripled_mask,      # Shape: (batch_size * 3, 1, seq_len, seq_len)
        }, False

    def _build_double_batch_multistep(self, inputs, user_spans, assistant_spans):
        """
        Double batch for multi-step prediction WITHOUT localized masks.

        Batch 1: LM loss (original sequences, normal causal)
        Batch 2: Predictors inserted (normal causal for both predictor and target embeddings)

        This is more efficient than triple batch (2x vs 3x memory) but target embeddings
        see full causal context instead of being isolated to their step.
        """
        batch_size = inputs["input_ids"].shape[0]
        device = inputs["input_ids"].device

        # === Find step boundaries (same as triple batch) ===
        all_step_boundaries = []
        num_pairs_per_sample = []

        for i in range(batch_size):
            user_span = None
            assistant_span = None
            if user_spans is not None:
                user_span = (user_spans[i][0].item(), user_spans[i][1].item())
            if assistant_spans is not None:
                assistant_span = (assistant_spans[i][0].item(), assistant_spans[i][1].item())

            # Find boundaries
            boundaries = self._find_step_boundaries(
                inputs["input_ids"][i],
                inputs["labels"][i],
                self.processing_class,
                user_span=user_span,
                assistant_span=assistant_span,
                num_steps=self.num_prediction_steps + 1
            )

            if boundaries is None or len(boundaries) < 2:
                # Fallback (same as triple batch)
                ids = inputs["input_ids"][i].tolist()
                sep_tokens = self.processing_class.encode("\n\n", add_special_tokens=False)
                actual_separators = 0
                for j in range(len(ids) - len(sep_tokens) + 1):
                    if ids[j:j+len(sep_tokens)] == sep_tokens:
                        actual_separators += 1
                max_possible_pairs = actual_separators if actual_separators > 0 else 1
                num_boundaries_to_create = min(self.num_prediction_steps + 1, max_possible_pairs + 1)
                last_token = self._last_token_index(
                    inputs["input_ids"][i:i+1],
                    inputs["labels"][i:i+1],
                    inputs["attention_mask"][i:i+1]
                )[0].item()

                if self.include_last_step_target:
                    # New behavior: include last_token position as final boundary
                    boundaries = [last_token * (j+1) // (num_boundaries_to_create + 1)
                                 for j in range(num_boundaries_to_create - 1)]
                    boundaries.append(last_token)  # Add exact last_token position
                else:
                    # Old behavior: omit final boundary
                    boundaries = [last_token * (j+1) // (num_boundaries_to_create + 1)
                                 for j in range(num_boundaries_to_create)]

            num_pairs = len(boundaries) - 1
            all_step_boundaries.append(boundaries)
            num_pairs_per_sample.append(num_pairs)

        # === BATCH 1: LM Loss (unchanged) ===
        batch1_input_ids = inputs["input_ids"]
        batch1_labels = inputs["labels"]

        # === BATCH 2: Insert predictors ===
        batch2_input_ids_list = []
        predictor_positions_list = []
        adjusted_target_positions_list = []  # NEW: track adjusted target positions

        for i in range(batch_size):
            boundaries = all_step_boundaries[i]
            num_pairs = num_pairs_per_sample[i]
            original_seq = inputs["input_ids"][i]

            new_seq_parts = []
            predictor_positions = []
            adjusted_target_positions = []  # Target positions after inserting predictors
            cumulative_offset = 0

            prev_pos = 0
            for step_idx in range(num_pairs):
                step_end = boundaries[step_idx]

                # Add tokens up to step_end
                new_seq_parts.append(original_seq[prev_pos:step_end+1])

                # Insert K predictor tokens
                predictor_ids = [
                    self.processing_class.convert_tokens_to_ids(f"<|predictor_{j+1}|>")
                    for j in range(self.step_jepa_predictors)
                ]
                new_seq_parts.append(torch.tensor(predictor_ids, device=device))

                # Track last predictor position
                predictor_pos = step_end + cumulative_offset + self.step_jepa_predictors
                predictor_positions.append(predictor_pos)

                # Track target position (step t+1) adjusted for predictors inserted so far
                if step_idx + 1 < len(boundaries):
                    target_pos_original = boundaries[step_idx + 1]
                    # Adjust for predictors inserted up to this point (step_idx+1 predictors)
                    target_pos_adjusted = target_pos_original + (step_idx + 1) * self.step_jepa_predictors
                    adjusted_target_positions.append(target_pos_adjusted)

                cumulative_offset += self.step_jepa_predictors
                prev_pos = step_end + 1

            # Add remaining tokens
            new_seq_parts.append(original_seq[prev_pos:])
            new_seq = torch.cat(new_seq_parts)

            batch2_input_ids_list.append(new_seq)
            predictor_positions_list.append(predictor_positions)
            adjusted_target_positions_list.append(adjusted_target_positions)

        # Pad batch2 sequences
        max_len_batch2 = max(seq.shape[0] for seq in batch2_input_ids_list)
        batch2_input_ids = torch.full(
            (batch_size, max_len_batch2),
            self.processing_class.pad_token_id,
            device=device
        )
        for i, seq in enumerate(batch2_input_ids_list):
            batch2_input_ids[i, :seq.shape[0]] = seq

        # Batch2 labels: all masked
        batch2_labels = torch.full((batch_size, max_len_batch2), -100, device=device)

        # === Pad both batches to same max_length ===
        max_len = max(batch1_input_ids.shape[1], batch2_input_ids.shape[1])

        def pad_to_length(tensor, target_len, pad_value):
            if tensor.shape[1] < target_len:
                padding = torch.full(
                    (tensor.shape[0], target_len - tensor.shape[1]),
                    pad_value,
                    device=device
                )
                return torch.cat([tensor, padding], dim=1)
            return tensor

        batch1_input_ids = pad_to_length(batch1_input_ids, max_len, self.processing_class.pad_token_id)
        batch1_labels = pad_to_length(batch1_labels, max_len, -100)
        batch2_input_ids = pad_to_length(batch2_input_ids, max_len, self.processing_class.pad_token_id)
        batch2_labels = pad_to_length(batch2_labels, max_len, -100)

        # === Create attention masks (DOUBLE batch, both normal causal) ===
        doubled_mask = torch.full((batch_size * 2, 1, max_len, max_len), float('-inf')).to(device)

        for i in range(batch_size):
            last_token = self._last_token_index(
                inputs["input_ids"][i:i+1],
                inputs["labels"][i:i+1],
                inputs["attention_mask"][i:i+1]
            )[0].item()

            # BATCH 1: Normal causal
            seq_len = last_token + 1
            doubled_mask[i, 0, :seq_len, :seq_len] = self._build_additive_mask(seq_len)

            # BATCH 2: Normal causal (with predictors)
            seq_len_with_pred = seq_len + num_pairs_per_sample[i] * self.step_jepa_predictors
            doubled_mask[i + batch_size, 0, :seq_len_with_pred, :seq_len_with_pred] = \
                self._build_additive_mask(seq_len_with_pred)

        # Store metadata
        self._all_step_boundaries = all_step_boundaries
        self._predictor_positions = predictor_positions_list
        self._adjusted_target_positions = adjusted_target_positions_list  # NEW
        self._num_pairs_per_sample = num_pairs_per_sample

        # Double batch (not triple)
        doubled_input_ids = torch.cat([batch1_input_ids, batch2_input_ids], dim=0)
        doubled_labels = torch.cat([batch1_labels, batch2_labels], dim=0)

        return {
            "input_ids": doubled_input_ids,      # Shape: (batch_size * 2, seq_len)
            "labels": doubled_labels,            # Shape: (batch_size * 2, seq_len)
            "attention_mask": doubled_mask,      # Shape: (batch_size * 2, 1, seq_len, seq_len)
        }, False

    def build_with_additive_mask(self, inputs):
        """
        Dispatcher for Step-JEPA batch construction.

        Routes to appropriate batch construction method based on num_prediction_steps
        and use_localized_masks config.
        """
        # Apply jepa_ratio dropout
        if self.jepa_ratio > 0.0:
            if torch.rand(1).item() > self.jepa_ratio:
                return {
                    "input_ids": inputs["input_ids"],
                    "labels": inputs["labels"],
                    "attention_mask": inputs["attention_mask"],
                }, True  # skip_jepa=True

        # Extract metadata
        user_spans = inputs.get("user_span", None)
        assistant_spans = inputs.get("assistant_span", None)

        # Route based on config
        if self.num_prediction_steps == 1:
            # Single-step: check if isolation masks enabled
            if self.use_localized_masks:
                # Original behavior (with view2 isolation)
                return self._build_double_batch(inputs, user_spans, assistant_spans)
            else:
                # New mode (no view2 isolation)
                return self._build_double_batch_no_isolation(inputs, user_spans, assistant_spans)
        else:
            # Multi-step: check if localized masks enabled
            if self.use_localized_masks:
                # Triple batch with localized step masks (current behavior)
                return self._build_triple_batch(inputs, user_spans, assistant_spans)
            else:
                # Double batch with normal causal masks (new mode)
                return self._build_double_batch_multistep(inputs, user_spans, assistant_spans)

    def forward(self, model, inputs):
        """
        Custom forward pass that handles all model calls.
        """
        # Main forward pass for language modeling
        assert self.additive_mask, "Additive mask must be used with Step-JEPA for this implementation"
        llm_inputs, skip_jepa = self.build_with_additive_mask(inputs)

        if self.debug == 7 and torch.cuda.current_device() == 0:
            torch.set_printoptions(threshold=float("inf"))
            torch.set_printoptions(linewidth=360)
            print(">>>input_ids<<<")
            print(llm_inputs["input_ids"])
            print(">>>labels<<<")
            print(llm_inputs["labels"])
            print(">>>attention_mask<<<")
            print(llm_inputs["attention_mask"])
            if self.additive_mask:
                print(">>>step1_end_pos<<<")
                print(self._step1_end_pos)
                print(">>>step2_end_pos<<<")
                print(self._step2_end_pos)
        if self.debug == 7:
            exit(0)
        if self.debug == 2 and torch.cuda.current_device() == 0:
            print("=====before:outputs=====")
            print("input_ids shapes:")
            print(llm_inputs["input_ids"].shape)
            print("labels shapes::")
            print(llm_inputs["labels"].shape)
            print("attention_mask shapes:")
            print(llm_inputs["attention_mask"].shape)

        with torch.set_grad_enabled(True):
            outputs = model(**llm_inputs, output_hidden_states=True)

        if self.debug == 2 and torch.cuda.current_device() == 0:
            print(f"=====outputs.loss.shape:{outputs.loss.shape}=====")
            print(f"=====outputs.hidden_states[-1].shape:{outputs.hidden_states[-1].shape}=====")
        

        if skip_jepa:
            jepa_hidden_states = None
        else:
            # Detect batch structure
            total_batch_size = llm_inputs["input_ids"].shape[0]
            original_batch_size = inputs["input_ids"].shape[0]

            if total_batch_size == original_batch_size * 2:
                # DOUBLE BATCH (either num_prediction_steps=1 OR use_localized_masks=False)
                if self.num_prediction_steps == 1:
                    # Single-step: extract second half only
                    jepa_hidden_states = outputs.hidden_states[-1][original_batch_size:original_batch_size*2]
                else:
                    # Multi-step without localized masks: keep second half for both predictor and target
                    jepa_hidden_states = outputs.hidden_states[-1][original_batch_size:original_batch_size*2]
            elif total_batch_size == original_batch_size * 3:
                # TRIPLE BATCH (use_localized_masks=True)
                jepa_hidden_states = outputs.hidden_states[-1]
            else:
                raise ValueError(f"Unexpected batch structure: total={total_batch_size}, original={original_batch_size}")

        if self.debug == 2 and torch.cuda.current_device() == 0:
            if jepa_hidden_states is not None:
                print(f"====={jepa_hidden_states.shape}=====") # shape: (batch_size, seq_len (include predictors), hidden_size)

        # Return all outputs needed for loss computation
        return {
            'main_outputs': outputs,
            'jepa_hidden_states': jepa_hidden_states,
        }
                
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute sft loss and jepa loss as regularization terms.
        Supports both single-step (N=1, double batch) and multi-step (N>1, triple batch) JEPA.
        """
        # Get all forward pass results
        forward_results = self.forward(model, inputs)

        # Extract main language modeling loss
        main_outputs = forward_results['main_outputs']
        lm_loss = main_outputs.loss

        # Compute representation similarity loss
        jepa_hidden_states = forward_results['jepa_hidden_states']

        # Branch based on num_prediction_steps
        if jepa_hidden_states is None:
            # skip_jepa = True
            jepa_loss = 0.0
            cosine_similarity = None
        elif self.num_prediction_steps == 1:
            # ===== DOUBLE BATCH: Single-step prediction (backward compatible) =====
            batch_size = jepa_hidden_states.shape[0]
            index_predictor = self._predictor_pos  # Position of last predictor token (after insertion)
            index_view2 = self._view2_end_pos  # Position of view2 end (after insertion)
            predictor_embedding = jepa_hidden_states[range(batch_size), index_predictor, :]
            view2_embedding = jepa_hidden_states[range(batch_size), index_view2, :]

            # Compute cosine similarity
            cosine_similarity = F.cosine_similarity(predictor_embedding, view2_embedding, dim=-1)
            if self.debug == 1 and torch.cuda.current_device() == 0:
                print(f"predictor_embedding.shape: {predictor_embedding.shape}, view2_embedding.shape: {view2_embedding.shape}")
                print(f"cosine_similarity.shape: {cosine_similarity.shape}")
                print(f"index_predictor (per sample): {index_predictor}")
                print(f"index_view2 (per sample): {index_view2}")
                print(f"cosine_similarity values (per sample): {cosine_similarity}")
                print(f"Are all index_predictor same? {torch.all(index_predictor == index_predictor[0])}")
                print(f"Are all index_view2 same? {torch.all(index_view2 == index_view2[0])}")
                print(f"Are all cosine_similarity same? {torch.all(cosine_similarity == cosine_similarity[0])}")

            # Compute JEPA loss
            if self.jepa_l2:
                jepa_loss = torch.linalg.norm(predictor_embedding - view2_embedding, ord=2, dim=-1).mean()
            elif self.jepa_mse:
                jepa_loss = torch.mean((predictor_embedding - view2_embedding) ** 2)
            elif self.infonce:
                predictor_norm = F.normalize(predictor_embedding, p=2, dim=1)
                view2_norm = F.normalize(view2_embedding, p=2, dim=1)
                cosine_sim = torch.mm(predictor_norm, view2_norm.T)
                infonce_logit = cosine_sim / 0.07  # temperature
                infonce_label = torch.arange(cosine_sim.size(0), device=cosine_sim.device)
                jepa_loss = F.cross_entropy(infonce_logit, infonce_label)
                if self.debug == 8:
                    print(cosine_sim.shape, infonce_logit.shape, infonce_label.shape, jepa_loss.shape)
                    exit(0)
            else:
                jepa_loss = 1.0 - torch.mean(cosine_similarity)
        else:
            # ===== MULTI-STEP PREDICTION =====
            # Detect if we're using triple batch (localized) or double batch (non-localized)
            total_batch_size = jepa_hidden_states.shape[0]
            original_batch_size = len(self._num_pairs_per_sample)

            if total_batch_size == original_batch_size * 3:
                # TRIPLE BATCH: localized masks enabled
                batch_size = original_batch_size
                batch2_hidden = jepa_hidden_states[batch_size:2*batch_size]  # Predictors
                batch3_hidden = jepa_hidden_states[2*batch_size:]             # Targets (localized)

                # Extract from separate batches (existing logic)
                total_loss = 0.0
                total_pairs = 0
                all_similarities = []

                for i in range(batch_size):
                    num_pairs = self._num_pairs_per_sample[i]
                    boundaries = self._all_step_boundaries[i]
                    predictor_positions = self._predictor_positions[i]

                    for pair_idx in range(num_pairs):
                        pred_pos = predictor_positions[pair_idx]
                        pred_emb = batch2_hidden[i, pred_pos, :]

                        target_pos = boundaries[pair_idx + 1]  # Original position
                        target_emb = batch3_hidden[i, target_pos, :]

                        sim = F.cosine_similarity(pred_emb.unsqueeze(0), target_emb.unsqueeze(0), dim=-1)
                        all_similarities.append(sim)

                        if self.jepa_l2:
                            total_loss += torch.linalg.norm(pred_emb - target_emb, ord=2)
                        elif self.jepa_mse:
                            total_loss += torch.mean((pred_emb - target_emb) ** 2)
                        elif self.infonce:
                            total_loss += (1.0 - sim)
                        else:
                            total_loss += (1.0 - sim)

                        total_pairs += 1

                jepa_loss = total_loss / total_pairs if total_pairs > 0 else 0.0

            elif total_batch_size == original_batch_size:
                # DOUBLE BATCH: localized masks disabled (use_localized_masks=False)
                batch_size = original_batch_size
                batch2_hidden = jepa_hidden_states  # Both predictor and target from same batch

                # Extract from SAME batch using adjusted positions
                total_loss = 0.0
                total_pairs = 0
                all_similarities = []

                for i in range(batch_size):
                    num_pairs = self._num_pairs_per_sample[i]
                    predictor_positions = self._predictor_positions[i]
                    adjusted_target_positions = self._adjusted_target_positions[i]  # NEW

                    for pair_idx in range(num_pairs):
                        pred_pos = predictor_positions[pair_idx]
                        pred_emb = batch2_hidden[i, pred_pos, :]

                        target_pos = adjusted_target_positions[pair_idx]  # Adjusted position
                        target_emb = batch2_hidden[i, target_pos, :]

                        sim = F.cosine_similarity(pred_emb.unsqueeze(0), target_emb.unsqueeze(0), dim=-1)
                        all_similarities.append(sim)

                        if self.jepa_l2:
                            total_loss += torch.linalg.norm(pred_emb - target_emb, ord=2)
                        elif self.jepa_mse:
                            total_loss += torch.mean((pred_emb - target_emb) ** 2)
                        elif self.infonce:
                            total_loss += (1.0 - sim)
                        else:
                            total_loss += (1.0 - sim)

                        total_pairs += 1

                jepa_loss = total_loss / total_pairs if total_pairs > 0 else 0.0
            else:
                raise ValueError(f"Unexpected batch structure in multi-step: total={total_batch_size}, original={original_batch_size}")

            # Stack similarities
            if len(all_similarities) > 0:
                cosine_similarity = torch.cat(all_similarities)
            else:
                cosine_similarity = None

        # Total loss
        total_loss = self.gamma * lm_loss + self.lbd * jepa_loss

        if self.debug == 2 and torch.cuda.current_device() == 0:
            if cosine_similarity is not None:
                print(lm_loss, self.lbd, torch.mean(cosine_similarity))
            else:
                print(lm_loss, self.lbd, "N/A (skip_jepa=True)")

        if self.debug == 1 or self.debug == 2:
            exit(0)

        if self.debug == 5 and torch.cuda.current_device() == 0:
            if jepa_hidden_states is not None and cosine_similarity is not None:
                cosine_sim_mean = torch.mean(cosine_similarity).item()
                cosine_sim_std = torch.std(cosine_similarity).item()
                cosine_sim_min = torch.min(cosine_similarity).item()
                cosine_sim_max = torch.max(cosine_similarity).item()
                print(f"llm_loss: {lm_loss}, jepa_loss: {jepa_loss}")
                print(f"  cosine_sim: mean={cosine_sim_mean:.4f}, std={cosine_sim_std:.4f}, min={cosine_sim_min:.4f}, max={cosine_sim_max:.4f}")
                if self.num_prediction_steps > 1:
                    print(f"  num_prediction_pairs: {total_pairs}")
                if self.num_prediction_steps == 1 and cosine_sim_std < 1e-6:
                    index_predictor = self._predictor_pos
                    index_view2 = self._view2_end_pos
                    print(f"  WARNING: All samples have same cosine_similarity! index_predictor={index_predictor}, index_view2={index_view2}")
            else:
                print(f"llm_loss: {lm_loss}, jepa_loss: {jepa_loss}")

        return (total_loss, main_outputs) if return_outputs else total_loss
