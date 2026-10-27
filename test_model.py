import torch
from inference.model import ModelArgs, Transformer

def test():
    args = ModelArgs()
    print("Initializing Neuron-500M-MoE with random weights on meta device...")
    
    with torch.device("meta"):
        model = Transformer(args)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n==========================================")
    print(f" ACTUAL PYTORCH MODEL TOTAL PARAMS: {total_params / 1e6:.3f}M")
    print(f"==========================================")

if __name__ == "__main__":
    test()
