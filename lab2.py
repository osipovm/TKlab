import itertools
import numpy as np

matrix_a = np.array([[1, 1, 1],
                     [1, 1, 0],
                     [1, 0, 1],
                     [0, 1, 1]])

matrix_b = np.array([[0, 1, 1, 1, 1, 0, 0],
                     [1, 1, 0, 1, 1, 1, 1],
                     [0, 0, 1, 1, 0, 1, 1],
                     [1, 0, 1, 0, 1, 1, 0]])

def generate_matrix(input_matrix, k):
    return np.hstack((np.eye(k, dtype=int), input_matrix))

def parity_check_matrix(input_matrix, total_n, num_k):
    return np.vstack((input_matrix, np.eye(total_n - num_k, dtype=int)))

def create_syndrome_table(H_matrix):
    return {str(row): [i] for i, row in enumerate(H_matrix)}

def fix_single_error(syndrome_dict, syndrome_vector, codeword):
    index = syndrome_dict.get(str(syndrome_vector))
    if index is None:
        print("Синдром не найден в таблице синдромов", '\n')
    else:
        codeword[index] ^= 1
    return codeword

def fix_double_error(syndrome_dict, syndrome_vector, codeword):
    indices = syndrome_dict.get(str(syndrome_vector), (-1, -1))
    for index in indices:
        if index != -1:
            codeword[index] ^= 1
    if indices[0] == -1:
        print("Синдром не найден в матрице синдромов")
    return codeword

def create_double_syndrome_table(H, n):
    table_syndromes = {str(row): [i] for i, row in enumerate(H)}
    mistakes = np.eye(n, dtype=int)

    for comb in itertools.combinations(range(n), 2):
        mistake_vector = (mistakes[comb[0]] + mistakes[comb[1]]) @ H % 2
        table_syndromes[str(mistake_vector)] = comb

    return table_syndromes

def first_part():
    print("Часть 1\n")
    total_n = 7
    num_k = 4

    gen_matrix = generate_matrix(matrix_a, num_k)
    print("Порождающая матрица G (7,4,3):", '\n', gen_matrix, '\n')

    check_matrix = parity_check_matrix(matrix_a, total_n, num_k)
    print("Проверочная матрица H:", '\n', check_matrix, '\n')

    syndrome_table = create_syndrome_table(check_matrix)
    print('Таблица синдромов для однократных ошибок:', syndrome_table, sep='\n')

    identity_matrix = np.eye(total_n, dtype=int)
    original_codeword = [1, 0, 1, 0]
    print('\nИсходное слово: ', np.array(original_codeword))
    transmitted_codeword = np.array(original_codeword) @ gen_matrix % 2
    print('Кодированное слово: ', transmitted_codeword)

    codeword_with_single_error = (transmitted_codeword + identity_matrix[5]) % 2
    print('Кодированное слово с одной ошибкой: ', codeword_with_single_error)
    syndrome_vector = codeword_with_single_error @ check_matrix % 2
    print('Полученный синдром: ', syndrome_vector)
    corrected_codeword = fix_single_error(syndrome_table, syndrome_vector, codeword_with_single_error)
    print('Исправленное слово: ', corrected_codeword)
    validation_corrected = corrected_codeword @ check_matrix % 2
    print("Проверка исправленного слова: ", validation_corrected, '\n')

    original_codeword = [1, 0, 1, 0]
    print('\nИсходное слово: ', np.array(original_codeword))
    transmitted_codeword = np.array(original_codeword) @ gen_matrix % 2
    print('Кодированное слово: ', transmitted_codeword)

    codeword_with_double_errors = (transmitted_codeword + identity_matrix[2] + identity_matrix[5]) % 2
    print('Кодированное слово с двумя ошибками: ', codeword_with_double_errors)
    syndrome_vector = codeword_with_double_errors @ check_matrix % 2
    print('Полученный синдром: ', syndrome_vector)
    corrected_codeword = fix_single_error(syndrome_table, syndrome_vector, codeword_with_double_errors)
    print('Исправленное слово (отличается от отправленного): ', corrected_codeword)
    validation_corrected = corrected_codeword @ check_matrix % 2
    print("Проверка исправленного слова: ", validation_corrected, '\n')

def second_part():
    print("Часть 2\n")
    total_n = 11
    num_k = 4

    gen_matrix = generate_matrix(matrix_b, num_k)
    print("Порождающая матрица G (11,4,5):", '\n', gen_matrix, '\n')

    check_matrix = parity_check_matrix(matrix_b, total_n, num_k)
    print("Проверочная матрица H:", '\n', check_matrix, '\n')

    syndrome_table = create_double_syndrome_table(check_matrix, total_n)
    print('Таблица синдромов:', syndrome_table, sep='\n')

    identity_matrix = np.eye(total_n, dtype=int)
    original_codeword = [1, 0, 1, 0]
    print('\nИсходное слово: ', np.array(original_codeword))
    transmitted_codeword = np.array(original_codeword) @ gen_matrix % 2
    print('Кодированное слово: ', transmitted_codeword)

    codeword_with_single_error = (transmitted_codeword + identity_matrix[5]) % 2
    print('Кодированное слово с одной ошибкой: ', codeword_with_single_error)
    syndrome_vector = codeword_with_single_error @ check_matrix % 2
    print('Полученный синдром: ', syndrome_vector)
    corrected_codeword = fix_single_error(syndrome_table, syndrome_vector, codeword_with_single_error)
    print('Исправленное слово: ', corrected_codeword)
    validation_corrected = corrected_codeword @ check_matrix % 2
    print("Проверка исправленного слова: ", validation_corrected, '\n')

    original_codeword = [1, 0, 1, 0]
    print('\nИсходное слово: ', np.array(original_codeword))
    transmitted_codeword = np.array(original_codeword) @ gen_matrix % 2
    print('Кодированное слово: ', transmitted_codeword)

    codeword_with_double_errors = (transmitted_codeword + identity_matrix[2] + identity_matrix[5]) % 2
    print('Кодированное слово с двумя ошибками: ', codeword_with_double_errors)
    syndrome_vector = codeword_with_double_errors @ check_matrix % 2
    print('Полученный синдром: ', syndrome_vector)
    corrected_codeword = fix_double_error(syndrome_table, syndrome_vector, codeword_with_double_errors)
    print('Исправленное слово: ', corrected_codeword)
    validation_corrected = corrected_codeword @ check_matrix % 2
    print("Проверка исправленного слова: ", validation_corrected, '\n')

    original_codeword = [0, 1, 1, 0]
    print('\nИсходное слово: ', np.array(original_codeword))
    transmitted_codeword = np.array(original_codeword) @ gen_matrix % 2
    print('Кодированное слово: ', transmitted_codeword)

    codeword_with_three_errors = (transmitted_codeword + identity_matrix[1] + identity_matrix[3] + identity_matrix[4]) % 2
    print('Кодированное слово с тремя ошибками: ', codeword_with_three_errors)
    syndrome_vector = codeword_with_three_errors @ check_matrix % 2
    print('Полученный синдром: ', syndrome_vector)
    corrected_codeword = fix_double_error(syndrome_table, syndrome_vector, codeword_with_three_errors)
    print('Исправленное слово (отличается от отправленного): ', corrected_codeword)
    validation_corrected = corrected_codeword @ check_matrix % 2
    print("Проверка исправленного слова: ", validation_corrected, '\n')

first_part()
second_part()
