import inspect, pymorphy2
if not hasattr(inspect, "getargspec"):
    def getargspec(func):
        fullspec = inspect.getfullargspec(func)
        return fullspec.args, fullspec.varargs, fullspec.varkw, fullspec.defaults
    inspect.getargspec = getargspec

morph = pymorphy2.MorphAnalyzer() 

def normalize_text(text):
    """
    Нормализует входной текст:
      - Разбивает строку на слова.
      - Для каждого слова получает его нормальную (лемматизированную) форму.
      - Объединяет слова обратно в строку.
    """
    words = text.split()  # Разбиваем текст на слова по пробелам
    normalized_words = [morph.parse(word)[0].normal_form for word in words]  # Получаем нормальную форму для каждого слова
    return " ".join(normalized_words)  # Возвращаем объединенную строку
