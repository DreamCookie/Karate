from transformers import pipeline
from model_loader import load_quantized_model
from config import MAX_NEW_TOKENS

model, tokenizer = load_quantized_model()


generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,       
    do_sample=False,                     
    early_stopping=True,
    eos_token_id=tokenizer.eos_token_id
)

def generate_solution(user_query: str, stage_mode: str):
    """
    Генерирует промпт, учитывая 5 основных этапов проектной деятельности, а также «восхождение» или «спуск» (stage_mode)
    param user_query запрос юзера
    param stage_mode 'восхождение' или 'спуск'
    """
    prompt = f"""
У нас есть 5 основных этапов:
1) Сформировать общую потребность.
2) Сформировать частные потребности.
3) Сформировать новые функции.
4) Сформировать новые свойства.
5) Сформировать новую функциональную структуру.
После чего формируется итоговое техническое решение.

При этом пользователь может выбрать «восхождение» или «спуск»:
- «восхождение» означает, что мы поднимаемся на более общий уровень проектной деятельности.
- «спуск» означает детализированную проработку (уточнение) для конкретного этапа.

Пользовательский запрос: {user_query}
Режим: {stage_mode}

Задача:
1) Используя информацию о 5 этапах, определи, на каком этапе мы находимся 
   (и/или нужна ли детальная проработка или обобщение)
2) Сформулируй пошаговое краткое решение или инструкцию, как двигаться дальше
3) Учитывай, что мы хотим ответить «до конца мысли», без искусственного обрыва ответа
   Если мысль короткая – заверши ответ
Обеспечь связный ответ, опираясь на этапы проектирования и выбранный режим («восхождение» или «спуск»)
"""
    result = generator(prompt, num_return_sequences=1)
    generated_text = result[0]["generated_text"]
    return generated_text
