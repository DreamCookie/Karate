from sentence_transformers import SentenceTransformer, util
import torch
from nlp_utils import normalize_text

# модель SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Глобальный словарь для хранения эмбеддингов категорий таксономии
racket_category_embeddings = {}

def build_racket_category_embeddings(taxonomy_dict):
    """
    Вычисляет эмбеддинги для всех ключей таксономии 
    Результаты сохраняются в глобальном словаре racket_category_embeddings
    """
    names = list(taxonomy_dict.keys())  # Получаем список категорий
    embeddings = model.encode(names, convert_to_tensor=True)  # Вычисляем эмбеддинги для списка имен
    for name, emb in zip(names, embeddings):
        racket_category_embeddings[name] = emb  # Сохраняем эмбеддинг для каждой категории

def find_best_racket_category(user_query, taxonomy_dict, top_k=1, threshold=0.5):
    """
    Вычисляет эмбеддинг запроса пользователя и ищет категорию таксономии с наибольшим косинусным сходством
    Если максимально найденное сходство ниже порога, возвращает None
    """
    user_query = user_query.lower()  # Приводим запрос к нижнему регистру
    query_emb = model.encode(user_query, convert_to_tensor=True)  # Вычисляем эмбеддинг запроса

    all_names = list(taxonomy_dict.keys())  # Получаем список всех категорий из таксономии
    all_embs = [racket_category_embeddings[name] for name in all_names]  # Извлекаем эмбеддинги для каждой категории

    similarities = util.cos_sim(query_emb, torch.stack(all_embs))[0]  # Вычисляем косинусное сходство
    best_val, best_idx = (torch.topk(similarities, k=top_k).values[0].item(),
                          torch.topk(similarities, k=top_k).indices[0].item())  # Находим наибольшее сходство и его индекс

    if best_val < threshold:  # Если максимальное сходство ниже порога, возвращаем None
        return None, best_val
    best_category_name = all_names[best_idx]  # Определяем имя категории с наибольшим сходством
    return best_category_name, best_val  # Возвращаем найденную категорию и значение сходства

def get_all_descendants(category_name, taxonomy_dict):
    """
    Рекурсивно собирает и возвращает список всех потомков (дочерних категорий) заданной категории из таксономии
    """
    descendants = []  # Инициализируем список потомков
    children = taxonomy_dict.get(category_name, {}).get("children", [])  # Получаем дочерние узлы
    for child in children:
        descendants.append(child)  # Добавляем дочернюю категорию в список
        descendants.extend(get_all_descendants(child, taxonomy_dict))  # Рекурсивно добавляем потомков дочерних категорий
    return descendants

def hierarchical_find_racket_category(user_query, taxonomy_dict, threshold=0.5):
    """
    Выполняет иерархический поиск наиболее подходящей категории по запросу:
      1. Вызывает find_best_racket_category для первичного поиска
      2. Если найденная категория является корневой (слишком общей, parent==None),
         просматривает всех ее потомков (через get_all_descendants) и пытается уточнить выбор,
         сравнивая нормализованные формы названий с нормализованным запросом.
    Возвращает уточненную категорию (если найдена) и значение сходства
    """
    best_cat, sim_val = find_best_racket_category(user_query, taxonomy_dict, threshold=threshold)
    if best_cat is None:
        return None, sim_val

    # Если найденная категория является корневой (не имеет родителя), уточняем выбор среди потомков
    if taxonomy_dict[best_cat]["parent"] is None:
        descendants = get_all_descendants(best_cat, taxonomy_dict)
        normalized_query = normalize_text(user_query)
        for d in descendants:
            norm_d = normalize_text(d)
            if norm_d in normalized_query or normalized_query in norm_d:
                return d, sim_val
    return best_cat, sim_val

def ascend_racket_category(category_name, taxonomy_dict):
    """
    Возвращает родительскую категорию для заданной категории
    Если категория не найдена в таксономии, возвращает None
    """
    if category_name not in taxonomy_dict:
        return None
    return taxonomy_dict[category_name]["parent"]

def descend_racket_category(category_name, taxonomy_dict):
    """
    Возвращает список непосредственных дочерних категорий для заданной категории
    Если категория не найдена, возвращает пустой список
    """
    if category_name not in taxonomy_dict:
        return []
    return taxonomy_dict[category_name].get("children", [])
