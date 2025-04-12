from generator import generate_solution

def main():
    print("Введите ваш запрос:")
    user_query = input().strip()
    
    print("Выберите режим ('восхождение' или 'спуск'):")
    stage_mode = input().strip().lower()
    
    print("\nГенерируется решение...")
    answer = generate_solution(user_query, stage_mode)
    
    print("\nИтоговое решение:")
    print(answer)

if __name__ == "__main__":
    main()