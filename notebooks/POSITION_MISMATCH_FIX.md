# 🔧 Position Mismatch Fix - `visualize_attention_around_high_entropy()`

## ❌ Vấn đề phát hiện

Bạn đúng! Có **position mismatch** giữa entropy positions và attention positions:

### Position Spaces:

1. **`entropies`**: 
   - Position 0 = **output token đầu tiên** 
   - KHÔNG bao gồm input tokens
   - Range: `[0, num_output_tokens)`

2. **`attention_weights`** (từ forward pass):
   - Bao gồm **cả input + output tokens**
   - Position 0 = BOS hoặc first input token
   - Range: `[0, input_length + output_length)`

3. **`tokens`**: 
   - Cũng bao gồm **cả input + output**
   - `tokens[0]` = first input token
   - `tokens[input_length]` = first output token

### Vấn đề cụ thể:

```python
# Code cũ (SAI):
pos = het['position']  # pos = 100 (trong OUTPUT space)
attn_window = attn[pos-10:pos+10, pos-10:pos+10]  # SAI! pos=100 nhưng trong FULL sequence space

# Kết quả: Lấy sai vùng attention!
```

## ✅ Giải pháp

### 1. Thêm parameter `input_length`

```python
def visualize_attention_around_high_entropy(
    attention_weights: tuple,
    entropies: List[Dict],
    tokens: List[str],
    input_length: int,  # ← NEW parameter
    ...
)
```

### 2. Offset positions đúng

```python
# Code mới (ĐÚNG):
output_pos = het['position']  # pos = 100 (OUTPUT space)
full_pos = output_pos + input_length  # pos = 240 (FULL sequence space, nếu input_length=140)

# Now use full_pos for attention indexing
start_pos = max(0, full_pos - context_window)
end_pos = min(len(tokens), full_pos + context_window + 1)
attn_window = attn[start_pos:end_pos, start_pos:end_pos]  # ĐÚNG!
```

### 3. Tăng context_window

```python
context_window: int = 30  # Tăng từ 10 → 30
```

## 📝 Updated Function Signature

```python
def visualize_attention_around_high_entropy(
    attention_weights: tuple,      # FROM forward pass (full sequence)
    entropies: List[Dict],           # OUTPUT positions only
    tokens: List[str],               # Full sequence tokens
    input_length: int,               # NEW: for position offset
    threshold_method: str = 'top_k',
    k: int = 5,
    layer_indices: Optional[List[int]] = None,
    head_idx: Optional[int] = None,
    context_window: int = 30,       # Increased from 10
    figsize: Optional[Tuple[int, int]] = None
):
    ...
```

## 🔄 How to Update Usage

### Trước (SAI):

```python
high_entropy_tokens, fig = visualize_attention_around_high_entropy(
    attention_weights=attentions_to_use,
    entropies=entropies,
    tokens=generated_tokens,
    threshold_method='top_k',
    k=5,
    context_window=10
)
```

### Sau (ĐÚNG):

```python
# Lấy input_length từ generation
input_length = inputs.input_ids.shape[1]  # or inputs['input_ids'].shape[1]

high_entropy_tokens, fig = visualize_attention_around_high_entropy(
    attention_weights=attentions_to_use,
    entropies=entropies,
    tokens=generated_tokens,
    input_length=input_length,  # ← ADD THIS
    threshold_method='top_k',
    k=5,
    context_window=30  # Increased
)
```

## 🎯 Verification

Function sẽ print thông tin debug:

```
================================================================================
Found 10 high entropy tokens (in OUTPUT)
Input length: 140, Total tokens: 2942
================================================================================

1. Output pos  242 (Full pos  382): '.

' (Entropy: 1.5690)
2. Output pos 2159 (Full pos 2299): 'So' (Entropy: 1.5661)
...
```

Giờ có thể verify:
- Output position: position trong entropy list
- Full position: position trong full sequence (cho attention indexing)
- Title cũng hiển thị cả 2: `(out=242, full=382)`

## 📊 Changes Summary

| Feature | Before | After |
|---------|--------|-------|
| **Position handling** | ❌ Dùng trực tiếp entropy pos | ✅ Offset bằng input_length |
| **Context window** | 10 tokens | 30 tokens |
| **Debug info** | Minimal | Full position info |
| **Title format** | `pos=X` | `out=X, full=Y` |

## 🚀 Full Example

```python
# Example 1: Generate & get entropy
messages = [{"role": "user", "content": question}]
inputs = tokenizer.apply_chat_template(messages, ...)

outputs = model.generate(**inputs, output_scores=True, ...)
input_length = inputs.input_ids.shape[1]  # SAVE THIS!

# Calculate entropy
scores = torch.stack(outputs.scores, dim=0)
entropies = calculate_token_entropy_transformers(scores, tokenizer, output_tokens)

# Example 2: Get attentions
full_sequence = outputs.sequences
forward_outputs = model(full_sequence, output_attentions=True)

attentions_to_use = forward_outputs.attentions
generated_tokens = [tokenizer.decode([tok]) for tok in outputs.sequences[0]]

# Visualize with correct offset
high_entropy_tokens, fig = visualize_attention_around_high_entropy(
    attention_weights=attentions_to_use,
    entropies=entropies,
    tokens=generated_tokens,
    input_length=input_length,  # ← CRITICAL!
    threshold_method='top_k',
    k=5,
    layer_indices=[0, 14, 27],
    context_window=30
)
plt.show()
```

## ✅ Checklist

- [x] Add `input_length` parameter
- [x] Offset entropy positions: `full_pos = output_pos + input_length`
- [x] Use `full_pos` for attention indexing
- [x] Increase `context_window` to 30
- [x] Add debug output showing both positions
- [x] Update title format to show both positions
- [x] Fix tick label boundary check
- [ ] Update usage examples in notebook cells

## 📁 Files

- **Fixed function**: `/Users/dungnh/coding-papers/jepa-math/notebooks/fix_visualize_attention_around_high_entropy.py`
- **This doc**: `/Users/dungnh/coding-papers/jepa-math/notebooks/POSITION_MISMATCH_FIX.md`

