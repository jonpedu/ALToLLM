#!/usr/bin/env python3
"""
Teste simples para verificar se o modelo ALToLLM carrega corretamente.
"""
import sys
import torch

def test_model_loading():
    """Testa o carregamento básico do modelo."""
    try:
        print("Iniciando teste de carregamento do modelo...")
        
        # Importações básicas
        from internvl.model.internvl_chat import ALToLLM
        print("✓ Importação do ALToLLM bem-sucedida")
        
        # Verificar se há GPU disponível
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✓ Dispositivo detectado: {device}")
        
        if device == 'cuda':
            print(f"✓ Memória GPU disponível: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print("✓ Teste de importação concluído com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
