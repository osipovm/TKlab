import random
import numpy as np

def generate_syndrome_table(matrix_h):
    return {str(row): [idx] for idx, row in enumerate(matrix_h)}

def get_syndrome_index(syndrome_table, syndrome):
    return syndrome_table.get(str(syndrome), [])

def encode_word(syndrome_table, syndrome, bit_array):
    pos = syndrome_table.get(str(syndrome))
    if pos is not None:
        bit_array[pos] ^= 1
    else:
        print("Синдрома нет в таблице синдромов")
    return bit_array

def create_hemming_matrix(par_r, extended=False):
    base_matrix = np.array([[int(bit) for bit in format(i, '0' + str(par_r) + 'b')] for i in range(2 ** par_r)])
    base_matrix = base_matrix[~(base_matrix.sum(axis=1) <= 1)]
    base_matrix = np.flip(base_matrix, 0)

    generator_matrix = np.hstack((np.eye(len(base_matrix), dtype=int), base_matrix))
    parity_check_matrix = np.vstack((base_matrix, np.eye(par_r, dtype=int)))

    if extended:
        parity_check_matrix = np.hstack((parity_check_matrix, np.zeros((parity_check_matrix.shape[0], 1), dtype=int)))
        parity_check_matrix = np.vstack((parity_check_matrix, np.ones((1, parity_check_matrix.shape[1]), dtype=int)))
        generator_matrix = np.hstack((generator_matrix, np.zeros((generator_matrix.shape[0], 1), dtype=int)))
        parity_col = generator_matrix.sum(axis=1) % 2
        generator_matrix[:, -1] = parity_col

    return generator_matrix, parity_check_matrix

def detect_and_fix_errors(gen_matrix, par_check_matrix, syndrome_table, par_r, is_extended):
    identity_matrix = np.eye(gen_matrix.shape[1], dtype=int)
    original_word = np.random.randint(0, 2, 2 ** par_r - par_r - 1)
    encoded_word = original_word @ gen_matrix % 2
    max_mistakes = 5 if is_extended else 4

    for error_count in range(1, max_mistakes):
        print("\nИсходное слово: ", original_word)
        print("Отправленное слово: ", encoded_word)

        error_indices = random.sample(range(0, identity_matrix.shape[0]), error_count)
        word_with_errors = encoded_word.copy()

        for idx in error_indices:
            word_with_errors ^= identity_matrix[idx]

        print("Слово с", error_count, "кратной ошибкой: ", word_with_errors)

        syndrome = word_with_errors @ par_check_matrix % 2
        print("Синдром: ", syndrome)

        corrected_word = encode_word(syndrome_table, syndrome, word_with_errors.copy())
        print("Исправленное: ", corrected_word)

        corrected_check = corrected_word @ par_check_matrix % 2
        print("Проверка исправленного слова с помощью умножения на матрицу H: ", corrected_check, '\n')

def basic_hemming():
    print("\nДля кода Хемминга \n")
    for r in range(2, 5):
        print("Параметр r = ", r, '\n')
        gen, par_check = create_hemming_matrix(r)
        syndrome_table = generate_syndrome_table(par_check)
        print("G:", gen, sep='\n')
        print("H:", par_check, sep='\n')
        print("Таблица синдромов: ", syndrome_table)
        detect_and_fix_errors(gen, par_check, syndrome_table, r, False)

def extended_hemming():
    print("\nДля расширенного кода Хемминга \n")
    for r in range(2, 5):
        print("Параметр r = ", r, '\n')
        gen, par_check = create_hemming_matrix(r, extended=True)
        syndrome_table = generate_syndrome_table(par_check)
        print("G:", gen, sep='\n')
        print("H:", par_check, sep='\n')
        print("Таблица синдромов: ", syndrome_table)
        detect_and_fix_errors(gen, par_check, syndrome_table, r, True)

if __name__ == '__main__':
    basic_hemming()
    extended_hemming()
