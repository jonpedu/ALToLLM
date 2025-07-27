#!/usr/bin/env python3
"""
Teste do script inference_altollm.py com debug adicional.
"""
import sys
import torch
import gc

# Limpar memória antes de começar
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

try:
    print("=== TESTE DE EXECUÇÃO INFERENCE_ALTOLLM ===")
    print(f"GPU disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Memória GPU livre: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
    
    # Importações
    print("\n1. Testando importações...")
    from transformers import AutoTokenizer
    from internvl.model.internvl_chat import ALToLLM
    print("✓ Importações bem-sucedidas")
    
    # Teste de carregamento com device map
    print("\n2. Tentando carregar modelo com device_map auto...")
    
    model_name_or_path = "yayafengzi/ALToLLM-8B"
    
    # Tentar carregar apenas o tokenizer primeiro
    print("2.1. Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
    print("✓ Tokenizer carregado")
    
    # Tentar carregar modelo com configurações de baixa memória
    print("2.2. Carregando modelo (pode demorar)...")
    model = ALToLLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    print("✓ Modelo carregado com sucesso!")
    print("✓ Todos os testes passaram!")
    
except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
