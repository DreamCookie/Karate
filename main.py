from taxonomy import racket_taxonomy
from nlp_utils import normalize_text
from search import build_racket_category_embeddings, hierarchical_find_racket_category, ascend_racket_category, descend_racket_category
from solution import auto_generate_components, form_technical_solution

def main():
    # Инициализируем эмбеддинги для таксономии
    build_racket_category_embeddings(racket_taxonomy)

    # Получаем запрос пользователя через консольный ввод
    print("Введите ваш запрос (например, 'охладить ручку'):")
    user_query = input().strip()
    
    # Запрашиваем режим работы: "восхождение" для получения родительского узла, "спуск" для дочерних категорий
    print("Выберите режим работы (введите 'восхождение' или 'спуск'):")
    mode = input().strip().lower()

    # Выполняем иерархический поиск по таксономии для определения наиболее подходящей категории
    matched_category, similarity_val = hierarchical_find_racket_category(user_query, racket_taxonomy, threshold=0.3)
    
    if matched_category is None:
        print(f"Нет подходящих категорий, максимальное сходство: {similarity_val:.4f}")
    else:
        print(f"Наиболее подходящая категория: {matched_category} (сходство: {similarity_val:.4f})")
        if mode == "восхождение":
            parent = ascend_racket_category(matched_category, racket_taxonomy)
            if parent:
                print(f"Родительский класс: {parent}")
            else:
                print("Это корневая категория, у неё нет родителя.")
        elif mode == "спуск":
            children = descend_racket_category(matched_category, racket_taxonomy)
            if children:
                print("Дочерние категории:", children)
            else:
                print("Нет дочерних категорий.")

        # Автоматически генерируем компоненты для итогового технического решения
        components = auto_generate_components(matched_category, racket_taxonomy)
        technical_solution = form_technical_solution(components)
        print("\nИтоговое техническое решение:")
        print(technical_solution)

if __name__ == "__main__":
    main()
