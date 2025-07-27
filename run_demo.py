import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    from transformers import AutoModel
    print("Transformers package is installed")
except ImportError:
    print("Transformers package is NOT installed")

print("All basic imports worked!")