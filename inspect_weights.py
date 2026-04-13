"""
Run this once before implementing model.py:
    python inspect_weights.py > tensor_shapes.txt

It prints every tensor name + shape from the DiT checkpoint,
which is required to correctly implement BitHumanExpressionModel.
"""
from safetensors import safe_open

weights_path = "bh-weights/bithuman-expression/Model_Lite/bithuman_expression_dit_1_3b.safetensors"

print(f"Opening {weights_path}\n")
with safe_open(weights_path, framework="pt") as f:
    keys = sorted(f.keys())
    print(f"Total tensors: {len(keys)}\n")
    for key in keys:
        t = f.get_tensor(key)
        print(f"{key}: {list(t.shape)}")
