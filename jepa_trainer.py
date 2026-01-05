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
        assert self.jepa_l2 + self.jepa_mse <= 1, "Only one of jepa_l2 and jepa_mse can be True."
        super().__init__(*args, **kwargs)
    
        
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
    
    def _find_step_boundaries(self, input_ids, labels, tokenizer):
        ids = input_ids.tolist()
        lab = labels.tolist()
        # print(f"lab: {lab}")
        # exit(0)
        # assistant_start: first non-masked label token
        assistant_start = None
        for i, x in enumerate(lab):
            if x != -100:
                assistant_start = i
                break
        if assistant_start is None:
            return None, None

        # assistant_end: last non-masked label token
        assistant_end = None
        for i in range(len(lab) - 1, -1, -1):
            if lab[i] != -100:
                assistant_end = i
                break
        if assistant_end is None or assistant_end <= assistant_start:
            return None, None

        sep_tokens = tokenizer.encode("\n\n", add_special_tokens=False)
        occ = []

        # search only within assistant span
        for i in range(assistant_start, assistant_end - len(sep_tokens) + 2):
            if ids[i:i+len(sep_tokens)] == sep_tokens:
                occ.append(i + len(sep_tokens) - 1)
                if len(occ) >= 2:
                    break

        if len(occ) < 2:
            return None, None

        return occ[0], occ[1]

    
    def build_with_additive_mask(self, inputs):
        """
        Override parent's method to support Step-JEPA masking.
        
        Follows the original finetune.py pattern:
        1. Insert K predictor tokens after Step 1
        2. DOUBLE the batch
        3. First half: Modified tokens + Normal causal mask
        4. Second half: Modified tokens + Step-JEPA isolation mask
        
        """
        # Apply jepa_ratio dropout (same as original)
        if self.jepa_ratio > 0.0:
            if torch.rand(1).item() > self.jepa_ratio:
                return {
                    "input_ids": inputs["input_ids"],
                    "labels": inputs["labels"],
                    "attention_mask": inputs["attention_mask"],
                }, True  # skip_jepa=True, means no jepa loss and compute sft loss only
        
        # Step-JEPA logic
        batch_size = inputs["input_ids"].shape[0]
        seq_length = inputs["input_ids"].shape[-1]
        device = inputs["input_ids"].device
        
        # Find step boundaries for each example
        step1_end_positions = []
        step2_end_positions = []
        
        for i in range(batch_size):
            step1_end, step2_end = self._find_step_boundaries(
                inputs["input_ids"][i],
                inputs["labels"][i],
                self.processing_class
            )
            
            if step1_end is None or step2_end is None:
                # Fall back to sequence middle if can't find boundaries
                last_token = self._last_token_index(
                    inputs["input_ids"][i:i+1],
                    inputs["labels"][i:i+1],
                    inputs["attention_mask"][i:i+1]
                )[0].item()
                step1_end = last_token // 3
                step2_end = (last_token * 2) // 3
            
            step1_end_positions.append(step1_end)
            step2_end_positions.append(step2_end)
        
        step1_end_pos = torch.tensor(step1_end_positions, device=device)
        step2_end_pos = torch.tensor(step2_end_positions, device=device)
        
        # Insert K predictor tokens after Step 1 for each example
        # Create new input_ids with predictor tokens inserted (for JEPA batch only)
        new_input_ids_with_pred = []
        for i in range(batch_size):
            step1_end = step1_end_pos[i].item()
            # Insert K predictor tokens after step1_end
            predictor_ids = [self.processing_class.convert_tokens_to_ids(f"<|predictor_{j+1}|>") 
                            for j in range(self.step_jepa_predictors)]
            
            new_seq = torch.cat([
                inputs["input_ids"][i, :step1_end+1],
                torch.tensor(predictor_ids, device=device), # system + user + step1 + \n\n + predictor tokens + step2+...
                inputs["input_ids"][i, step1_end+1:]
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
            step1_end = step1_end_pos[i].item()
            step2_end = step2_end_pos[i].item()
            
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
            # Step-JEPA isolation mask
            
            # Calculate positions after inserting K predictor tokens
            predictor_start = step1_end + 1
            predictor_end = predictor_start + self.step_jepa_predictors - 1
            step2_start = predictor_end + 1
            step2_end_adjusted = step2_end + self.step_jepa_predictors
            seq_len_with_pred = last_token + 1 + self.step_jepa_predictors
            
            # - Everything before Step 2: normal causal
            mask[i + batch_size, 0, :step2_start, :step2_start] = self._build_additive_mask(step2_start)
            # - Step 2: isolated (can only see itself)
            mask[i + batch_size, 0, step2_start:step2_end_adjusted+1, step2_start:step2_end_adjusted+1] = \
                self._build_additive_mask(step2_end_adjusted - step2_start + 1)
            # - Step 3+: normal causal (can see everything)
            if step2_end_adjusted + 1 < seq_len_with_pred:
                mask[i + batch_size, 0, step2_end_adjusted+1:seq_len_with_pred, :seq_len_with_pred] = \
                    self._build_additive_mask(seq_len_with_pred)[step2_end_adjusted+1:seq_len_with_pred, :seq_len_with_pred]
        
        # Store positions for later use in compute_loss
        # These are for the SECOND HALF of the doubled batch
        self._step1_end_pos = step1_end_pos
        self._step2_end_pos = step2_end_pos + self.step_jepa_predictors  # Adjusted for inserted tokens
        self._predictor_pos = step1_end_pos + self.step_jepa_predictors  # Last predictor token
        # if self.debug == 5 and torch.cuda.current_device() == 0:
        #     print(f">>>step1_end_pos<<< {self._step1_end_pos}")
        #     print(f">>>step2_end_pos<<< {self._step2_end_pos}")
        #     print(f">>>predictor_pos<<< {self._predictor_pos}")
        return {
            "input_ids": doubled_input_ids,      # Shape: (batch_size * 2, seq_len)
            "labels": doubled_labels,            # Shape: (batch_size * 2, seq_len)
            "attention_mask": mask,              # Shape: (batch_size * 2, 1, seq_len, seq_len)
        }, False

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
            batch_size = llm_inputs["input_ids"].shape[0] // 2
            jepa_hidden_states = outputs.hidden_states[-1][batch_size: batch_size * 2] # shape: (batch_size, seq_len (include predictors), hidden_size)

        if self.debug == 2 and torch.cuda.current_device() == 0:
            print(f"====={jepa_hidden_states.shape}=====") # shape: (batch_size, seq_len (include predictors), hidden_size)
       
        # Return all outputs needed for loss computation
        return {
            'main_outputs': outputs,
            'jepa_hidden_states': jepa_hidden_states,
        }
                
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute sft loss and jepa loss as regularization terms.
        """
        # Get all forward pass results
        forward_results = self.forward(model, inputs)
        num_items = inputs["input_ids"].shape[0]
        # Extract main language modeling loss
        main_outputs = forward_results['main_outputs']
        lm_loss = main_outputs.loss

        # Compute representation similarity loss
        jepa_hidden_states = forward_results['jepa_hidden_states']
        
        # Get embeddings (using predictor position and step2 position)
        if jepa_hidden_states is not None:
            # Use batch_size from jepa_hidden_states (already sliced from doubled batch)
            batch_size = jepa_hidden_states.shape[0]
            index_predictor = self._predictor_pos  # Position of last predictor token (after insertion)
            index_step2 = self._step2_end_pos  # Position of step2 end (after insertion)
            predictor_embedding = jepa_hidden_states[range(batch_size), index_predictor, :]
            step2_embedding = jepa_hidden_states[range(batch_size), index_step2, :]
            
            # Compute cosine similarity
            cosine_similarity = F.cosine_similarity(predictor_embedding, step2_embedding, dim=-1)
            if self.debug == 1 and torch.cuda.current_device() == 0:
                print(f"predictor_embedding.shape: {predictor_embedding.shape}, step2_embedding.shape: {step2_embedding.shape}")
                print(f"cosine_similarity.shape: {cosine_similarity.shape}")
                print(f"index_predictor (per sample): {index_predictor}")
                print(f"index_step2 (per sample): {index_step2}")
                print(f"cosine_similarity values (per sample): {cosine_similarity}")
                print(f"Are all index_predictor same? {torch.all(index_predictor == index_predictor[0])}")
                print(f"Are all index_step2 same? {torch.all(index_step2 == index_step2[0])}")
                print(f"Are all cosine_similarity same? {torch.all(cosine_similarity == cosine_similarity[0])}")
    
            # Compute total loss
            if self.jepa_l2:
                jepa_loss = torch.linalg.norm(predictor_embedding - step2_embedding, ord=2, dim=-1).mean()
            elif self.jepa_mse:
                jepa_loss = torch.mean((predictor_embedding - step2_embedding) ** 2)
            elif self.infonce:
                predictor_norm = F.normalize(predictor_embedding, p=2, dim=1)
                step2_norm = F.normalize(step2_embedding, p=2, dim=1)
                cosine_sim = torch.mm(predictor_norm, step2_norm.T)
                infonce_logit = cosine_sim / 0.07  # temperature
                infonce_label = torch.arange(cosine_sim.size(0), device=cosine_sim.device)
                jepa_loss = F.cross_entropy(infonce_logit, infonce_label)
                if self.debug == 8:
                    print(cosine_sim.shape, infonce_logit.shape, infonce_label.shape, jepa_loss.shape)
                    exit(0)
            else:
                jepa_loss = 1.0 - torch.mean(cosine_similarity)
        else:
            jepa_loss = 0.0
            cosine_similarity = None

        total_loss = self.gamma * lm_loss + self.lbd * jepa_loss

        if self.debug == 2 and torch.cuda.current_device() == 0:
            if cosine_similarity is not None:
                print(lm_loss, self.lbd, torch.mean(cosine_similarity))
            else:
                print(lm_loss, self.lbd, "N/A (skip_jepa=True)")

        if self.debug == 1 or self.debug == 2:
            exit(0)

        if self.debug == 5 and torch.cuda.current_device() == 0:
            if jepa_hidden_states is not None:
                cosine_sim_mean = torch.mean(cosine_similarity).item()
                cosine_sim_std = torch.std(cosine_similarity).item()
                cosine_sim_min = torch.min(cosine_similarity).item()
                cosine_sim_max = torch.max(cosine_similarity).item()
                print(f"llm_loss: {lm_loss.float():.4f}, jepa_loss: {jepa_loss.float():.4f}")
                print(f"  cosine_sim: mean={cosine_sim_mean:.4f}, std={cosine_sim_std:.4f}, min={cosine_sim_min:.4f}, max={cosine_sim_max:.4f}")
                if cosine_sim_std < 1e-6:
                    print(f"  WARNING: All samples have same cosine_similarity! index_predictor={index_predictor}, index_step2={index_step2}")
            else:
                print(f"llm_loss: {lm_loss.float():.4f}, jepa_loss: {jepa_loss.float():.4f}")

        return (total_loss, main_outputs) if return_outputs else total_loss
