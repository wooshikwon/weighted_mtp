
import os
import torch
from pathlib import Path
from omegaconf import OmegaConf
from transformers import LlamaConfig, LlamaModel
from weighted_mtp.models.value_model import ValueModel
from weighted_mtp.models.lora import get_hf_lora_state_dict

def reproduce():
    print("Starting reproduction script...")
    
    # 1. Setup Dummy Config and Model
    config_dict = {
        "models": {
            "value_model": {
                "path": "dummy-path",
                "dtype": "float32",
                "checkpoint_path": "dummy_checkpoint.pt"
            }
        },
        "training": {
            "value_head": {"type": "mlp", "dropout": 0.1}
        }
    }
    config = OmegaConf.create(config_dict)
    
    llama_config = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128
    )
    
    backbone = LlamaModel(llama_config)
    # Create ValueModel
    # We need to manually create it as from_pretrained requires a real path usually, 
    # but we can instantiate directly.
    from weighted_mtp.models.value_head import create_value_head
    value_head = create_value_head(32, "mlp", 0.1)
    
    value_model = ValueModel(backbone, value_head, llama_config)
    
    # Apply LoRA
    lora_config = {
        "rank": 4,
        "alpha": 8.0,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"] # Llama uses q_proj, v_proj usually
    }
    value_model.apply_lora(lora_config)
    value_model.lora_enabled = True # This is set by apply_lora usually
    
    print("Model created with LoRA.")
    
    # 2. Simulate Saving
    # Logic from run_critic.py save_value_model_checkpoint
    
    # Mock unwrapped model (it's already unwrapped here)
    unwrapped = value_model
    
    checkpoint_path = Path("dummy_checkpoint.pt")
    
    checkpoint = {
        "checkpoint_type": "hf_lora",
        "lora_state_dict": get_hf_lora_state_dict(unwrapped.backbone),
        "value_head_state_dict": unwrapped.value_head.state_dict(),
        "lora_config": unwrapped.lora_config,
        "base_model_path": "dummy-base-path", # Mocking config.models.value_model.path
        "epoch": 1.0,
        "train_metrics": {},
        "val_metrics": {},
        "config": OmegaConf.to_container(config, resolve=True),
    }
    
    print(f"Saving checkpoint to {checkpoint_path}...")
    torch.save(checkpoint, checkpoint_path)
    print("Checkpoint saved.")
    
    # 3. Simulate Loading
    print("Loading checkpoint...")
    
    # We need to mock from_pretrained to avoid loading real model from disk
    # We can patch ValueModel.from_pretrained or just test _load_from_lora_checkpoint directly
    # But to test full flow, let's try to use from_checkpoint but mock the inner from_pretrained
    
    original_from_pretrained = ValueModel.from_pretrained
    
    def mock_from_pretrained(model_path, **kwargs):
        print(f"Mock loading base model from {model_path}")
        # Return a new fresh model
        backbone = LlamaModel(llama_config)
        value_head = create_value_head(32, "mlp", 0.1)
        model = ValueModel(backbone, value_head, llama_config)
        if kwargs.get("use_lora"):
            model.apply_lora(kwargs.get("lora_config"))
        return model
        
    ValueModel.from_pretrained = mock_from_pretrained
    
    try:
        loaded_model = ValueModel.from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            device="cpu",
            base_model_path_override="dummy-override-path"
        )
        print("Model loaded successfully!")
        
        # Verify LoRA weights
        # Check if state dicts match
        original_lora = get_hf_lora_state_dict(value_model.backbone)
        loaded_lora = get_hf_lora_state_dict(loaded_model.backbone)
        
        keys_match = original_lora.keys() == loaded_lora.keys()
        print(f"LoRA keys match: {keys_match}")
        
        # Check a value
        if keys_match and len(original_lora) > 0:
            k = list(original_lora.keys())[0]
            val_match = torch.allclose(original_lora[k], loaded_lora[k])
            print(f"LoRA value match for {k}: {val_match}")
            
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore
        ValueModel.from_pretrained = original_from_pretrained
        if checkpoint_path.exists():
            os.remove(checkpoint_path)

if __name__ == "__main__":
    reproduce()
