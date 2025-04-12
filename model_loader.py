from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME
import torch

def load_quantized_model(model_name=MODEL_NAME):
    """
    окружение должно быть настроено для использования CUDA и bitsandbytes... на винде пока работает не очень
    """
    print(f"Загрузка модели: {model_name} на GPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    model.eval()  
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_quantized_model()
    print("Модель успешно загружена на GPU")
