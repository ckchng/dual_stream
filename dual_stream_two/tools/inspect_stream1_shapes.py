
import torch
import sys
import os

# Add the project root to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bisenetv2 import BiSeNetv2Dual, BiSeNetv2DualHTlastlayer

def print_shape_hook(module, input, output):
    # Get the name of the module if possible, or just print the class name
    class_name = module.__class__.__name__
    if isinstance(output, tuple):
        # For SemanticBranch which might return aux heads
        print(f"{class_name} output: {[o.shape for o in output if isinstance(o, torch.Tensor)]}")
    else:
        print(f"{class_name} output: {output.shape}")

def inspect_stream1():
    # Create the model
    # model = BiSeNetv2DualHTlastlayer(num_class=2, n_channel=3)
    model = BiSeNetv2Dual(num_class=2, n_channel=3)
    model.eval()

    # Create dummy input (Batch Size 1, 3 Channels, 512x1024 or whatever size you use)
    # Using 512x1024 as a standard example size
    x1 = torch.randn(1, 3, 192, 416)
    x2 = torch.randn(1, 3, 192, 416)

    print("Input shape:", x1.shape)
    print("-" * 30)
    print("Stream 1 Layer Shapes:")

    # Hook into Detail Branch 1 layers
    # DetailBranch is a Sequential, so we can iterate over its children
    print("\n--- Detail Branch 1 ---")
    for i, layer in enumerate(model.detail_branch1):
        layer.register_forward_hook(lambda m, i, o, idx=i: print(f"DetailBranch Layer {idx} ({m.__class__.__name__}): {o.shape}"))

    # Hook into Semantic Branch 1 stages
    print("\n--- Semantic Branch 1 ---")
    model.semantic_branch1.stage1to2.register_forward_hook(lambda m, i, o: print(f"Stage 1-2 (StemBlock): {o.shape}"))
    model.semantic_branch1.stage3.register_forward_hook(lambda m, i, o: print(f"Stage 3: {o.shape}"))
    model.semantic_branch1.stage4.register_forward_hook(lambda m, i, o: print(f"Stage 4: {o.shape}"))
    model.semantic_branch1.stage5_1to4.register_forward_hook(lambda m, i, o: print(f"Stage 5 (1-4): {o.shape}"))
    model.semantic_branch1.stage5_5.register_forward_hook(lambda m, i, o: print(f"Stage 5 (5) (ContextEmbedding): {o.shape}"))

    # Hook into BGA Layer 1
    print("\n--- BGA Layer 1 ---")
    model.bga_layer1.register_forward_hook(lambda m, i, o: print(f"BGA Layer: {o.shape}"))

    # Run forward pass
    with torch.no_grad():
        output = model(x1, x2)
        print(f"Final Model Output (Interpolated): {output.shape}")

if __name__ == "__main__":
    inspect_stream1()
