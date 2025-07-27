import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

# Configurar para usar menos memória
torch.backends.cuda.max_split_size_mb = 256
torch.cuda.empty_cache()

# Verificar memória GPU disponível
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU total memory: {gpu_memory:.2f} GB")
    
    # Verificar memória livre
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    cached = torch.cuda.memory_reserved(0) / 1024**3
    free = gpu_memory - cached
    print(f"Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB, Free: {free:.2f} GB")
else:
    device = torch.device("cpu")
    print("CUDA não está disponível, usando CPU")

# Tentar carregamento básico primeiro
print("Tentando carregar apenas o tokenizer...")
path = 'yayafengzi/ALToLLM-8B'

try:
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    print("✓ Tokenizer carregado com sucesso!")
except Exception as e:
    print(f"✗ Erro ao carregar tokenizer: {e}")
    exit(1)

# Agora tentar carregar o modelo com máxima otimização
print("Tentando carregar o modelo...")

try:
    from internvl.model.internvl_chat import ALToLLM
    
    # Configuração mais agressiva de baixa memória
    model = ALToLLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="balanced_low_0",  # Distribuir de forma balanceada com prioridade baixa
        max_memory={0: "7GB", "cpu": "16GB"},  # Limitar uso de GPU a 7GB
        offload_folder="./temp_offload",  # Pasta temporária para offload
        offload_buffers=True
    ).eval()
    
    print("✓ Modelo carregado com sucesso!")
    
    # Verificar memória após carregamento
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"Após carregamento - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

except Exception as e:
    print(f"✗ Erro ao carregar modelo: {e}")
    import traceback
    traceback.print_exc()
    
    # Tentar CPU fallback
    print("Tentando fallback para CPU...")
    try:
        model = ALToLLM.from_pretrained(
            path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu"
        ).eval()
        print("✓ Modelo carregado em CPU!")
    except Exception as e2:
        print(f"✗ Erro também em CPU: {e2}")
        exit(1)

print("Script executado com sucesso!")
