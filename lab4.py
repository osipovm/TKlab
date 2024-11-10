import numpy as np
import random

core_matrix = np.array([
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
])

def matrix_construction(C):
    gen_matrix = np.hstack((np.eye(12, dtype=int), C))
    check_matrix = np.vstack((np.eye(12, dtype=int), C))
    return gen_matrix, check_matrix

def induce_errors(data, gen_mat, num_errors):
    coded_word = data @ gen_mat % 2
    print(f"\nИсходное слово: {data}")
    print(f"Закодированное слово: {coded_word}")

    error_loc = random.sample(range(coded_word.shape[0]), num_errors)
    error_vec = np.zeros(coded_word.shape[0], dtype=int)

    for loc in error_loc:
        error_vec[loc] = 1

    corrupted = (coded_word + error_vec) % 2
    print(f"Слово с {num_errors} ошибками: {corrupted}")
    return corrupted


def locate_errors(broken_word, chk_matrix, ref_matrix):
    trouble = broken_word @ chk_matrix % 2

    def find_error_vector(trouble, matrix):
        for idx in range(len(matrix)):
            adjusted = (trouble + matrix[idx]) % 2
            if np.sum(adjusted) <= 2:
                error_vector = np.zeros(len(trouble), dtype=int)
                error_vector[idx] = 1
                return np.hstack((adjusted, error_vector))
        return None

    if np.sum(trouble) <= 3:
        return np.hstack((trouble, np.zeros(len(trouble), dtype=int)))

    error_vector = find_error_vector(trouble, ref_matrix)
    if error_vector is not None:
        return error_vector

    syndrome_extended = trouble @ ref_matrix % 2
    if np.sum(syndrome_extended) <= 3:
        return np.hstack((np.zeros(len(trouble), dtype=int), syndrome_extended))

    return find_error_vector(syndrome_extended, ref_matrix)

def amend_errors(input_word, modified, chk_matrix, ref_matrix, gen_matrix):
    detect_vec = locate_errors(modified, chk_matrix, ref_matrix)

    if detect_vec is None:
        print("Ошибка обнаружена, исправить невозможно!")
        return

    fixed_word = (modified + detect_vec) % 2
    print("Исправленное слово:", fixed_word)

    coded_original = input_word @ gen_matrix % 2
    if not np.array_equal(coded_original, fixed_word):
        print("Ошибка в декодированном сообщении!")

def partOne():
    print("\nЧасть 1")

    gen_matrix, chk_matrix = matrix_construction(core_matrix)
    print(f"Порождающая матрица G:\n{gen_matrix}\nПроверочная матрица H:\n{chk_matrix}")

    word_sample = np.array([i % 2 for i in range(len(gen_matrix))])

    for num_errors in range(5):
        corrupted_word = induce_errors(word_sample, gen_matrix, num_errors)
        amend_errors(word_sample, corrupted_word, chk_matrix, core_matrix, gen_matrix)
        print('')

def RM_code_construct(r_lvl, depth):
    if 0 < r_lvl < depth:
        main_section = RM_code_construct(r_lvl, depth - 1)
        sub_section = RM_code_construct(r_lvl - 1, depth - 1)
        return np.hstack([np.vstack([main_section, np.zeros((len(sub_section), len(main_section.T)), int)]),
                          np.vstack([main_section, sub_section])])
    elif r_lvl == 0:
        return np.ones((1, 2 ** depth), dtype=int)
    elif r_lvl == depth:
        primary = RM_code_construct(depth - 1, depth)
        sec = np.zeros((1, 2 ** depth), dtype=int)
        sec[0][-1] = 1
        return np.vstack([primary, sec])

def check_matrix_lvl(level, depth):
    H_base = np.array([[1, 1], [1, -1]])
    outcome = np.kron(np.eye(2 ** (depth - level)), H_base)
    outcome = np.kron(outcome, np.eye(2 ** (level - 1)))
    return outcome

def analyze_RM_errors(data_word, gen_matrix, error_count, depth):
    damaged = induce_errors(data_word, gen_matrix, error_count)

    for i in range(len(damaged)):
        if damaged[i] == 0:
            damaged[i] = -1

    weights = [damaged @ check_matrix_lvl(1, depth)]
    for level in range(2, depth + 1):
        weights.append(weights[-1] @ check_matrix_lvl(level, depth))

    highest = weights[0][0]
    pos = -1

    for j in range(len(weights)):
        for k in range(len(weights[j])):
            if abs(weights[j][k]) > abs(highest):
                pos = k
                highest = weights[j][k]

    occurrences = 0
    for j in range(len(weights)):
        for k in range(len(weights[j])):
            if abs(weights[j][k]) == abs(highest):
                occurrences += 1
            if occurrences > 1:
                print("Ошибка не может быть исправлена.\n")
                return

    corrected_output = list(map(int, list(('{' + f'0:0{depth}b' + '}').format(pos))))
    if highest > 0:
        corrected_output.append(1)
    else:
        corrected_output.append(0)

    print(f"Исправленное слово: {np.array(corrected_output[::-1])}")

def partTwo():
    print("\nЧасть 2")

    depth = 3
    print(f"\nПорождающая матрица Рида-Маллера (1,3): \n{RM_code_construct(1, depth)}\n")

    word_sample = np.array([i % 2 for i in range(len(RM_code_construct(1, depth)))])

    for error_count in range(1, 3):
        analyze_RM_errors(word_sample, RM_code_construct(1, depth), error_count, depth)

    depth = 4
    print(f"\nПорождающая матрица Рида-Маллера (1,4): \n{RM_code_construct(1, depth)}\n")

    word_sample = np.array([i % 2 for i in range(len(RM_code_construct(1, depth)))])

    for error_count in range(1, 5):
        analyze_RM_errors(word_sample, RM_code_construct(1, depth), error_count, depth)

if __name__ == '__main__':
    partOne()
    partTwo()
