import torch
import torch.nn as nn

# Set seed for reproducibility
torch.manual_seed(42)

B, IN, H, OUT = 2, 4, 3, 2

# Hardcoded input (same as C++)
input_data = torch.tensor([
    # Batch 0
    [0.5, -0.3, 1.2, -0.8],
    # Batch 1
    [0.1, 0.9, -0.5, 0.7]
], dtype=torch.float32)

print("Input:")
print(input_data)

# Create layers
l1 = nn.Linear(IN, H, bias=True)
l2 = nn.Linear(H, OUT, bias=True)

# Initialize weights with normal(0, 1) and bias with 0
nn.init.normal_(l1.weight, mean=0.0, std=1.0)
nn.init.zeros_(l1.bias)
nn.init.normal_(l2.weight, mean=0.0, std=1.0)
nn.init.zeros_(l2.bias)

# Forward pass
h = l1(input_data)
a = torch.relu(h)
y = l2(a)

print("\nAfter L1 (h):")
print(h)

print("\nAfter ReLU (a):")
print(a)

print("\nFinal Output (y):")
print(y)

# Print first 10 weights from l1 for verification
print("\nFirst 10 weights from l1:")
print(l1.weight.flatten()[:10])

# Additional verification
print("\n--- Statistics ---")
print(f"Output mean: {y.mean().item():.6f}")
print(f"Output std: {y.std().item():.6f}")
print(f"Output min: {y.min().item():.6f}")
print(f"Output max: {y.max().item():.6f}")