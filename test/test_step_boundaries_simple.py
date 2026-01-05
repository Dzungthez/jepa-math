import json

# Load a few samples
with open('../datasets/gsm8k_step_jepa.jsonl', 'r') as f:
    samples = []
    for i, line in enumerate(f):
        if i >= 10:  # Test 10 samples
            break
        samples.append(json.loads(line))

print("=" * 80)
print("Testing step boundaries detection in raw text")
print("=" * 80)

for idx, sample in enumerate(samples):
    messages = sample['messages']
    
    # Get assistant content
    assistant_content = None
    for msg in messages:
        if msg['role'] == 'assistant':
            assistant_content = msg['content']
            break
    
    if assistant_content is None:
        continue
    
    # Find first and second \n\n in assistant content
    first_nn = assistant_content.find('\n\n')
    second_nn = assistant_content.find('\n\n', first_nn + 2) if first_nn != -1 else -1
    
    print(f"\nSample {idx}:")
    print(f"  First \\n\\n at position: {first_nn}")
    print(f"  Second \\n\\n at position: {second_nn}")
    
    if first_nn != -1:
        # Show context around first \n\n
        start = max(0, first_nn - 30)
        end = min(len(assistant_content), first_nn + 50)
        context1 = assistant_content[start:end]
        print(f"  Context around first \\n\\n:")
        print(f"    {repr(context1)}")
    
    if second_nn != -1:
        # Show context around second \n\n
        start = max(0, second_nn - 30)
        end = min(len(assistant_content), second_nn + 50)
        context2 = assistant_content[start:end]
        print(f"  Context around second \\n\\n:")
        print(f"    {repr(context2)}")

# Now check if positions are the same across samples
print("\n" + "=" * 80)
print("Checking if positions are the same across samples")
print("=" * 80)

first_positions = []
second_positions = []

for sample in samples:
    messages = sample['messages']
    assistant_content = None
    for msg in messages:
        if msg['role'] == 'assistant':
            assistant_content = msg['content']
            break
    
    if assistant_content:
        first_nn = assistant_content.find('\n\n')
        second_nn = assistant_content.find('\n\n', first_nn + 2) if first_nn != -1 else -1
        first_positions.append(first_nn)
        second_positions.append(second_nn)

print(f"\nFirst \\n\\n positions: {first_positions}")
print(f"Second \\n\\n positions: {second_positions}")
print(f"All first positions same? {all(x == first_positions[0] for x in first_positions if x != -1)}")
print(f"All second positions same? {all(x == second_positions[0] for x in second_positions if x != -1)}")

# Check if they're in the same relative position (after tokenization might align)
print("\n" + "=" * 80)
print("Checking relative positions (first few characters of assistant content)")
print("=" * 80)

for idx, sample in enumerate(samples[:5]):
    messages = sample['messages']
    assistant_content = None
    for msg in messages:
        if msg['role'] == 'assistant':
            assistant_content = msg['content']
            break
    
    if assistant_content:
        first_nn = assistant_content.find('\n\n')
        print(f"Sample {idx}: First 100 chars: {repr(assistant_content[:100])}")
        print(f"  First \\n\\n at: {first_nn}")
        if first_nn != -1:
            print(f"  Text before first \\n\\n: {repr(assistant_content[:first_nn])}")

