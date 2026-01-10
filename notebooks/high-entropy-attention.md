## 📚 So sánh 3 hàm Attention Visualization

### Visual Overview:

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1️⃣ visualize_attention() - 2D Heatmaps                              │
├─────────────────────────────────────────────────────────────────────┤
│  Input: layers_heads = [(0, 0), (14, 0), (27, 27)]                 │
│                                                                     │
│  Output:  [Layer 0 Head 0]  [Layer 14 Head 0]  [Layer 27 Head 27] │
│           ┌───────────┐     ┌───────────┐      ┌───────────┐      │
│           │ █░░░      │     │ █░░░      │      │ █░░░      │      │
│           │ ██░░      │     │ ██░░      │      │ ██░░      │      │
│           │ ░██░      │     │ █░██      │      │ █░██      │      │
│           │ ░░██      │     │ █░░█      │      │ █░░█      │      │
│           └───────────┘     └───────────┘      └───────────┘      │
│                                                                     │
│  Best for: Detailed analysis của specific heads                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 2️⃣ visualize_attention_patterns() - 1D Bar Charts                   │
├─────────────────────────────────────────────────────────────────────┤
│  Input: layers_heads = [(0, 0), (14, 0), (27, 0)]                  │
│         pattern_type = 'mean' or 'specific_token'                  │
│                                                                     │
│  Output:  [L0H0 Mean]      [L14H0 Mean]     [L27H0 Mean]          │
│           ║                ║                ║                      │
│           ║ █              ║ ███            ║ ███                  │
│           ║ ██             ║ █              ║ █                    │
│           ║ ███            ║ ██             ║ ██                   │
│           ╚═══            ╚═══            ╚═══                     │
│           0 5 10 15        0 5 10 15       0 5 10 15               │
│                                                                     │
│  Best for: Finding important tokens, aggregated patterns           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 3️⃣ visualize_attention_layers() - Multi-Layer Grid (Paper Style)    │
├─────────────────────────────────────────────────────────────────────┤
│  Input: layer_indices = [0, 1, 2, 9, 14, 18, 23, 27]               │
│         head_idx = None (average all heads)                        │
│                                                                     │
│  Output:                                                            │
│    [L0 Avg]    [L1 Avg]    [L2 Avg]    [L9 Avg]                   │
│    ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐                    │
│    │ █░░░  │  │ ██░░  │  │ ██░░  │  │ █░░░  │  (Local pattern)   │
│    │ ██░░  │  │ ░██░  │  │ ░██░  │  │ █░░░  │                    │
│    │ ░██░  │  │ ░░██  │  │ ░░██  │  │ █░░░  │                    │
│    └───────┘  └───────┘  └───────┘  └───────┘                    │
│                                                                     │
│   [L14 Avg]   [L18 Avg]   [L23 Avg]   [L27 Avg]                   │
│    ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐                    │
│    │ █░░░  │  │ █░░░  │  │ █░░░  │  │ █░░░  │  (Attention sink)  │
│    │ █░░░  │  │ █░░░  │  │ █░░░  │  │ █░░░  │                    │
│    │ █░░░  │  │ █░░░  │  │ █░░░  │  │ █░░░  │                    │
│    └───────┘  └───────┘  └───────┘  └───────┘                    │
│                                                                     │
│  Best for: Cross-layer comparison, paper-style visualization       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1️⃣ `visualize_attention()` - Detailed Heatmaps
**Mục đích:** Hiển thị FULL attention matrix (2D heatmap) cho từng (layer, head) cụ thể

**Input:**
- `layers_heads`: List of (layer_idx, head_idx) - VD: [(0, 0), (14, 0), (27, 27)]
- Mỗi subplot hiển thị 1 head cụ thể

**Output:** 
- Grid của heatmaps 2D
- Mỗi cell = attention score từ query token i → key token j
- Có token labels trên axes

**Use case:**
- Phân tích chi tiết attention matrix
- So sánh specific heads
- Debug attention patterns

---

### 2️⃣ `visualize_attention_patterns()` - 1D Pattern Analysis
**Mục đích:** Summarize attention thành 1D pattern (bar chart)

**Input:**
- `layers_heads`: List of (layer_idx, head_idx) 
- `pattern_type`: 
  - 'mean' → Average attention nhận được bởi mỗi position
  - 'max' → Max attention nhận được
  - 'specific_token' → Attention FROM một token cụ thể

**Output:**
- Bar charts (1D)
- Mỗi bar = aggregated attention score cho 1 position

**Use case:**
- Tìm tokens nào nhận nhiều attention (important tokens)
- Phân tích attention FROM một token quan trọng
- So sánh attention distribution across layers

---

### 3️⃣ `visualize_attention_layers()` - Multi-Layer Overview (Paper Style)
**Mục đích:** Hiển thị attention patterns ACROSS nhiều layers (giống Figure 2 trong paper)

**Input:**
- `layer_indices`: List of layers - VD: [0, 1, 2, 9, 16, 23, 27]
- `head_idx`: 
  - None → Average across ALL heads (recommended)
  - 0, 1, 2... → Specific head

**Output:**
- Grid của heatmaps
- Mỗi subplot = 1 layer
- Consistent colorbar (để so sánh giữa layers)
- Colormap RdBu_r (red=high, blue=low)

**Use case:**
- Phân tích evolution của attention patterns qua layers
- Tìm "attention sink" phenomenon
- Paper-style visualization
- So sánh local vs global attention

---

### 📊 Quick Comparison Table

| Feature | visualize_attention | visualize_attention_patterns | visualize_attention_layers |
|---------|----------------------|-------------------------------|------------------------------|
| **Plot Type** | 2D Heatmap | 1D Bar Chart | 2D Heatmap Grid |
| **Input** | (layer, head) pairs | (layer, head) pairs | Layer indices only |
| **Head Selection** | Must specify | Must specify | Can average all heads |
| **Output** | Full attention matrix | Aggregated pattern | Multiple layers |
| **Best For** | Detailed analysis | Finding important tokens | Cross-layer comparison |
| **Paper Style** | ❌ | ❌ | ✅ |

---

### 💡 Workflow Recommendation

**Step 1:** Dùng `visualize_attention_layers()` để có overview
→ Tìm layers thú vị (local pattern, attention sink, etc.)

**Step 2:** Dùng `visualize_attention()` để phân tích chi tiết layers/heads cụ thể
→ Xem full attention matrix

**Step 3:** Dùng `visualize_attention_patterns()` để tìm important tokens
→ Tokens nào được attend nhiều? Token X attend vào đâu?


### 🎯 What to Look For in High Entropy Attention Patterns

**Yellow cross-hairs** = Position của high entropy token

**Patterns thú vị:**

1. **Broad attention distribution** (nhiều màu đỏ scattered)
   → Model đang "thinking", xem xét nhiều context

2. **Narrow attention** (ít màu đỏ, concentrated)
   → Model dựa vào specific tokens để decide

3. **Attention to earlier tokens** (màu đỏ ở bên trái)
   → Model "looking back" để lấy information

4. **Self-attention spike** (màu đỏ ở yellow cross)
   → Token attend strongly to itself

5. **Different patterns across layers:**
   - Early layers: Usually local attention
   - Middle layers: May show reasoning patterns
   - Late layers: Final decision making

**Use cases:**
- Debug model reasoning: Tại sao model uncertain ở token này?
- Find decision points: Token nào là "turning points" trong reasoning?
- Understand errors: Khi model sai, attention patterns như thế nào?
