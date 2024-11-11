import numpy as np
import math
from itertools import combinations, product

def generate_binary_matrix(cols):
    return list(product([0, 1], repeat=cols))

def calculate_f_value(vector, indices):
    return np.prod([(vector[idx] + 1) % 2 for idx in indices])

def create_vector(indices, num_cols):
    if len(indices) == 0:
        return np.ones(2 ** num_cols, dtype=int)
    return [calculate_f_value(binary_vector, indices) for binary_vector in generate_binary_matrix(num_cols)]

def generate_combinations(num_cols, r):
    return [subset for subset_size in range(r + 1) for subset in combinations(range(num_cols), subset_size)]

def compute_rm_matrix_size(r, m):
    return sum(math.comb(m, i) for i in range(r + 1))

def build_rm_matrix(r, m):
    size = compute_rm_matrix_size(r, m)
    matrix = np.zeros((size, 2 ** m), dtype=int)
    for row, subset in enumerate(generate_combinations(m, r)):
        matrix[row] = create_vector(subset, m)
    return matrix

def sort_for_decoding(m, r):
    index_combinations = list(combinations(range(m), r))
    index_combinations.sort(key=len)
    return np.array(index_combinations, dtype=int)

def create_vector_H(indices, m):
    return [binary_vector for binary_vector in generate_binary_matrix(m) if calculate_f_value(binary_vector, indices) == 1]

def get_complement(indices, m):
    return [i for i in range(m) if i not in indices]

def calculate_f_with_t(binary_vector, indices, t):
    return np.prod([(binary_vector[j] + t[j] + 1) % 2 for j in indices])

def create_vector_with_t(indices, m, t):
    if len(indices) == 0:
        return np.ones(2 ** m, dtype=int)
    return [calculate_f_with_t(binary_vector, indices, t) for binary_vector in generate_binary_matrix(m)]

def majoritarian_decoding(received_word, r, m, size):
    word = received_word.copy()
    decoded_vector = np.zeros(size, dtype=int)
    max_weight = 2 ** (m - r - 1) - 1
    index = 0

    for i in range(r, -1, -1):
        for indices in sort_for_decoding(m, i):
            max_count = 2 ** (m - i - 1)
            zero_count, one_count = 0, 0
            complement = get_complement(indices, m)

            for t in create_vector_H(indices, m):
                V = create_vector_with_t(complement, m, t)
                c = np.dot(word, V) % 2
                zero_count += (c == 0)
                one_count += (c == 1)

            if zero_count > max_weight and one_count > max_weight:
                return None

            if zero_count > max_count:
                decoded_vector[index] = 0
            elif one_count > max_count:
                decoded_vector[index] = 1
                word = (word + create_vector(indices, m)) % 2
            index += 1

    return decoded_vector

def generate_word_with_errors(G, error_count):
    u = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    print("Исходное сообщение:", u)
    encoded_word = np.dot(u, G) % 2
    error_positions = np.random.choice(len(encoded_word), size=error_count, replace=False)
    encoded_word[error_positions] = (encoded_word[error_positions] + 1) % 2
    return encoded_word

def run_single_error_experiment(G):
    error_word = generate_word_with_errors(G, 1)
    print("Слово с одной ошибкой:", error_word)
    decoded_word = majoritarian_decoding(error_word, 2, 4, len(G))
    if decoded_word is None:
        print("\nНеобходима повторная отправка сообщения")
    else:
        print("Исправленное слово:", decoded_word)
        result = np.dot(decoded_word, G) % 2
        print("Результат умножения исправленного слова на матрицу G:", result)

def run_double_error_experiment(G):
    error_word = generate_word_with_errors(G, 2)
    print("Слово с двумя ошибками:", error_word)
    decoded_word = majoritarian_decoding(error_word, 2, 4, len(G))
    if decoded_word is None:
        print("\nНеобходима повторная отправка сообщения")
    else:
        print("Исправленное слово:", decoded_word)
        result = np.dot(decoded_word, G) % 2
        print("Результат умножения исправленного слова на матрицу G:", result)

G_matrix = build_rm_matrix(2, 4)
print("Порождающая матрица G:\n", G_matrix)
run_single_error_experiment(G_matrix)
run_double_error_experiment(G_matrix)
