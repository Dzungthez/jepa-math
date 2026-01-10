# 🔧 Fix: Hiển thị nhiều token labels hơn trên trục X/Y

## ❌ Vấn đề hiện tại

Nhìn hình visualization, chỉ có **vài tokens được hiển thị** trên Query/Key Position axes (`my`, `1`, `plus`, `is`, etc.)

### Nguyên nhân:

1. **Context window nhỏ**: `context_window=15` → chỉ ±15 tokens (30 total)
2. **Tick step lớn**: Với window=30, `tick_step=5` → chỉ hiển thị 6 labels
3. **Font size nhỏ**: `fontsize=6` khó đọc

## ✅ Giải pháp đã implement

### 1. Tăng context_window

**Cell 30 - Usage code:**
```python
context_window=50,  # Show ±50 tokens (100 total) - INCREASED from 15!
```

### 2. Update tick logic (trong function definition)

**Cell chứa `visualize_attention_around_high_entropy` function:**

Tìm section này:
```python
# OLD CODE (BAD):
if window_size <= 20:
    tick_step = 2
elif window_size <= 40:
    tick_step = 5
else:
    tick_step = max(1, window_size // 10)  # Only ~10 labels!
```

Thay bằng:
```python
# NEW CODE (GOOD):
# Adaptive tick step based on window size
if window_size <= 20:
    tick_step = 1  # Show every token
elif window_size <= 40:
    tick_step = 2  # Show every 2nd token
elif window_size <= 60:
    tick_step = 3  # Show every 3rd token  
elif window_size <= 100:
    tick_step = 5  # Show every 5th token (~20 labels)
else:
    tick_step = max(1, window_size // 20)  # At least 20 labels
```

### 3. Increase font size & rotation

```python
# OLD:
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=6)
ax.set_yticklabels(tick_labels, fontsize=6)

# NEW:
ax.set_xticklabels(tick_labels, rotation=60, ha='right', fontsize=7)  # +1 fontsize, +15° rotation
ax.set_yticklabels(tick_labels, fontsize=7)
```

### 4. Shorter token display

```python
# OLD:
tick_labels = [tokens[start_pos + i][:10] for i in tick_positions ...]

# NEW:
tick_labels = [tokens[start_pos + i][:8] for i in tick_positions ...]  # Truncate to 8 chars
```

## 📊 Kết quả mong đợi

| Metric | Before | After |
|--------|--------|-------|
| **Tokens shown** | 30 (±15) | 100 (±50) |
| **Labels on axis** | ~6 labels | ~20 labels |
| **Font size** | 6pt | 7pt |
| **Rotation** | 45° | 60° |
| **Readability** | ❌ Poor | ✅ Good |

## 🔍 Manual Fix Steps

### Step 1: Tìm function definition

Search for `def visualize_attention_around_high_entropy` trong notebook

### Step 2: Update tick logic

Trong function, tìm section:
```python
# Tick labels (show every nth token)
if window_size <= 20:
    tick_step = 2
...
```

Thay thế bằng code mới ở trên.

### Step 3: Update usage call

Trong cell gọi function (Cell 30), đã update:
```python
high_entropy_tokens, fig = visualize_attention_around_high_entropy(
    ...
    context_window=50,  # ← INCREASED!
    ...
)
```

## 📈 Với window=50, sẽ thấy:

- **100 tokens total** (50 trước + 50 sau high entropy token)
- **~20 tick labels** trên mỗi trục (mỗi 5th token)
- **Labels rõ ràng hơn** với fontsize=7 và rotation=60°

## 🎯 Expected Output Example

```
Visualizing:
Position 1457 → Full window from pos 1407 to 1507 (100 tokens)
X-axis labels: tok1407, tok1412, tok1417, ..., tok1457 (YELLOW), ..., tok1502, tok1507
Y-axis labels: same

Instead of just: my, 1, plus, is (only 4-5 labels)
```

## ✅ Verification

After fixing, you should see:
1. **More context** around high entropy token
2. **~15-20 labels** on each axis instead of 5-6
3. **Clearer token text** at larger fontsize
4. **Yellow crosshairs** clearly marking the high entropy token in center

## 📝 Quick Fix Code Block

Copy this function code to replace in notebook:

```python
# In visualize_attention_around_high_entropy function, replace tick section:

            # Tick labels - show more labels for better readability
            if window_size <= 20:
                tick_step = 1
            elif window_size <= 40:
                tick_step = 2
            elif window_size <= 60:
                tick_step = 3
            elif window_size <= 100:
                tick_step = 5
            else:
                tick_step = max(1, window_size // 20)
            
            tick_positions = list(range(0, window_size, tick_step))
            tick_labels = [tokens[start_pos + i][:8] for i in tick_positions if start_pos + i < len(tokens)]
            
            ax.set_xticks(tick_positions[:len(tick_labels)])
            ax.set_xticklabels(tick_labels, rotation=60, ha='right', fontsize=7)
            ax.set_yticks(tick_positions[:len(tick_labels)])
            ax.set_yticklabels(tick_labels, fontsize=7)
```

Then in usage cell, set:
```python
context_window=50  # or even 70 for more context
```


