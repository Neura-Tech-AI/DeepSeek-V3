import torch
from model import ModelArgs, Transformer

def test_fwd():
    print("Initializing Neuron-500M-MoE on CPU for forward pass...")
    args = ModelArgs()
    
    # Model ko bfloat16 mein cast karte hain taki weights aur inputs match ho jayein
    torch.set_default_dtype(torch.float32) # default input type
    model = Transformer(args)
    model.to(torch.bfloat16)  # Model weights ko bf16 mein convert kiya
    model.eval()
    
    # Dummy token IDs (Batch size = 2, Sequence length = 16)
    tokens = torch.randint(0, args.vocab_size, (2, 16))
    
    print(f"Input tokens shape: {tokens.shape}")
    
    with torch.no_grad():
        logits = model(tokens)
        
    print(f"Output logits shape: {logits.shape}")
    print("SUCCESS: Forward pass completed without any errors!")

if __name__ == "__main__":
    test_fwd()
