def auto_generate_components(matched_category, taxonomy_dict):

    lower_cat = matched_category.lower()
    if "охлажд" in lower_cat:
        parent = taxonomy_dict.get(matched_category, {}).get("parent", "")
        if parent == "Ручка":
            return {"Охлаждение ручки": matched_category}
        else:
            return {"Охлаждение": matched_category}
    elif "информир" in lower_cat:
        return {"Информирование о натяжении струн": matched_category}
    elif "демпфир" in lower_cat:
        return {"Демпфирование колебаний": matched_category}
    else:
        return {matched_category: matched_category}

def form_technical_solution(components_dict):
    """
    Формирует итоговое техническое решение в виде текстового описания,
    объединяя выбранные компоненты из словаря
    """
    solution_description = "Предлагаемая функциональная структура теннисной ракетки:\n"
    for function, component in components_dict.items():
        solution_description += f"- Для функции '{function}' используется компонент: {component}\n"
    return solution_description
