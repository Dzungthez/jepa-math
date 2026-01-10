def visualize_attention_around_high_entropy(
    attention_weights: tuple,
    entropies: List[Dict],
    tokens: List[str],
    input_length: int,  # NEW: Length of input tokens (to offset entropy positions)
    threshold_method: str = 'top_k',
    k: int = 5,
    layer_indices: Optional[List[int]] = None,
    head_idx: Optional[int] = None,
    context_window: int = 30,  # Increased from 10 to 30
    figsize: Optional[Tuple[int, int]] = None
):
    """
    Visualize attention patterns xung quanh high entropy tokens
    
    IMPORTANT: entropy positions are relative to OUTPUT tokens only (0-indexed),
               but attention_weights and tokens include INPUT tokens.
               So we need to offset by input_length!
    
    Args:
        attention_weights: Tuple of attention tensors FROM FORWARD PASS
                          (includes both input and output tokens)
        entropies: List of entropy dicts (positions are OUTPUT-only, 0-indexed)
        tokens: List of ALL token strings (input + output)
        input_length: Length of input tokens (for position offset)
        threshold_method: 'top_k', 'percentile', hoặc 'std'
        k: Number of high entropy tokens to analyze
        layer_indices: Layers to visualize (None = auto select)
        head_idx: Head to visualize (None = average all heads)
        context_window: Number of tokens before/after to show (default 30)
        figsize: Figure size
    
    Returns:
        high_entropy_tokens: List of identified high entropy tokens
        fig: matplotlib figure
    """
    # Get high entropy tokens
    high_entropy_tokens = get_high_entropy_tokens(
        entropies,
        threshold_method=threshold_method,
        k=k
    )
    
    if len(high_entropy_tokens) == 0:
        print("No high entropy tokens found!")
        return high_entropy_tokens, None
    
    print(f"\n{'='*80}")
    print(f"Found {len(high_entropy_tokens)} high entropy tokens (in OUTPUT)")
    print(f"Input length: {input_length}, Total tokens: {len(tokens)}")
    print(f"{'='*80}\n")
    
    for i, het in enumerate(high_entropy_tokens[:10], 1):  # Show top 10
        output_pos = het['position']
        full_pos = output_pos + input_length
        print(f"{i}. Output pos {output_pos:4d} (Full pos {full_pos:4d}): '{het['token']}' (Entropy: {het['entropy']:.4f})")
    
    # Auto-select layers if not provided
    if layer_indices is None:
        num_layers = len(attention_weights)
        layer_indices = [0, num_layers // 2, num_layers - 1]
    
    # Create subplots for each high entropy token
    n_tokens = min(len(high_entropy_tokens), 5)  # Max 5 tokens
    n_layers = len(layer_indices)
    
    if figsize is None:
        figsize = (6 * n_layers, 4 * n_tokens)
    
    fig, axes = plt.subplots(n_tokens, n_layers, figsize=figsize)
    
    # Handle single row/column cases
    if n_tokens == 1 and n_layers == 1:
        axes = np.array([[axes]])
    elif n_tokens == 1:
        axes = axes.reshape(1, -1)
    elif n_layers == 1:
        axes = axes.reshape(-1, 1)
    
    for token_idx, het in enumerate(high_entropy_tokens[:n_tokens]):
        # CRITICAL FIX: offset entropy position by input_length
        output_pos = het['position']  # Position in output (0-indexed)
        full_pos = output_pos + input_length  # Position in full sequence
        
        # Calculate window in full sequence space
        start_pos = max(0, full_pos - context_window)
        end_pos = min(len(tokens), full_pos + context_window + 1)
        window_size = end_pos - start_pos
        
        for layer_plot_idx, layer_idx in enumerate(layer_indices):
            ax = axes[token_idx, layer_plot_idx]
            
            # Get attention
            attn = attention_weights[layer_idx][0].detach().cpu().numpy()
            
            if head_idx is not None:
                attn = attn[head_idx]
            else:
                attn = attn.mean(axis=0)  # Average across heads
            
            # Extract window
            attn_window = attn[start_pos:end_pos, start_pos:end_pos]
            
            # Plot
            im = ax.imshow(attn_window, cmap='RdBu_r', aspect='auto', interpolation='nearest')
            
            # Highlight high entropy token position (relative to window)
            het_rel_pos = full_pos - start_pos
            ax.axhline(het_rel_pos, color='yellow', linewidth=2, linestyle='--', alpha=0.7)
            ax.axvline(het_rel_pos, color='yellow', linewidth=2, linestyle='--', alpha=0.7)
            
            # Title
            head_str = f"H{head_idx}" if head_idx is not None else "Avg"
            ax.set_title(f"'{het['token']}' (out={output_pos}, full={full_pos})\nL{layer_idx} {head_str}", 
                        fontsize=9, fontweight='bold')
            
            # Labels
            ax.set_xlabel('Key Position', fontsize=8)
            ax.set_ylabel('Query Position', fontsize=8)
            
            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Tick labels - show more labels for better readability
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
            
            tick_positions = list(range(0, window_size, tick_step))
            tick_labels = [tokens[start_pos + i][:8] for i in tick_positions if start_pos + i < len(tokens)]
            
            ax.set_xticks(tick_positions[:len(tick_labels)])
            ax.set_xticklabels(tick_labels, rotation=60, ha='right', fontsize=7)  # Increased fontsize & rotation
            ax.set_yticks(tick_positions[:len(tick_labels)])
            ax.set_yticklabels(tick_labels, fontsize=7)
    
    plt.tight_layout()
    fig.suptitle(f'Attention Patterns Around High Entropy Tokens (Top {n_tokens}, window=±{context_window})', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.subplots_adjust(top=0.97)
    
    return high_entropy_tokens, fig

