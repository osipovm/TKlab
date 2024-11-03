import numpy as np

def swap_rows(matrix, row1, row2):
    matrix[[row1, row2]] = matrix[[row2, row1]]

def remove_zero_rows(matrix):
    temp_matrix = np.copy(matrix)
    zero_row_indices = []
    for row in range(temp_matrix.shape[0]):
        if not np.any(temp_matrix[row][:]) > 0:
            zero_row_indices.append(row)
    temp_matrix = np.delete(temp_matrix, zero_row_indices, axis=0)
    return temp_matrix

def generate_subsets(lst):
    return recursive_subsets([], sorted(lst))

def recursive_subsets(current_set, lst):
    if lst:
        return recursive_subsets(current_set, lst[1:]) + recursive_subsets(current_set + [lst[0]], lst[1:])
    return [current_set]

def row_echelon_form(matrix):
    echelon_matrix = np.copy(matrix)
    row_index = 0
    pivot_found = False
    for col in range(echelon_matrix.shape[1]):
        for row in range(row_index, echelon_matrix.shape[0]):
            if np.any(echelon_matrix[:, col]) > 0:
                if echelon_matrix[row, col] == 1:
                    if not pivot_found:
                        swap_rows(echelon_matrix, row, row_index)
                        pivot_found = True
                    else:
                        echelon_matrix[row] = (echelon_matrix[row] + echelon_matrix[row_index]) % 2
        if pivot_found:
            row_index += 1
            pivot_found = False

    echelon_matrix = remove_zero_rows(echelon_matrix)
    return echelon_matrix

def reduced_row_echelon_form(matrix):
    rref_matrix = row_echelon_form(matrix)
    for row in range(rref_matrix.shape[0] - 1, 0, -1):
        pivot_index = 0
        for col in range(rref_matrix.shape[1]):
            if rref_matrix[row][col] == 1:
                pivot_index = col
                break
        for prev_row in range(0, row):
            if rref_matrix[prev_row][pivot_index] == 1:
                rref_matrix[prev_row] = (rref_matrix[prev_row] + rref_matrix[row]) % 2
    return rref_matrix

def create_check_matrix(generator_matrix):
    transformed_matrix = reduced_row_echelon_form(generator_matrix)
    num_rows = generator_matrix.shape[0]
    num_cols = generator_matrix.shape[1]
    pivot_positions = []
    for row in range(num_rows):
        for col in range(num_cols):
            if transformed_matrix[row][col] == 1:
                pivot_positions.append(col)
                break
    transformed_matrix = np.delete(transformed_matrix, pivot_positions, axis=1)
    new_cols = num_cols - len(pivot_positions)
    identity_matrix = np.eye(new_cols)
    check_matrix = np.zeros((new_cols + num_rows, new_cols), dtype=int)
    row_index_transformed = 0
    row_index_identity = 0
    for row in range(len(check_matrix)):
        if row in pivot_positions:
            check_matrix[row] = transformed_matrix[row_index_transformed]
            row_index_transformed += 1
        else:
            check_matrix[row] = identity_matrix[row_index_identity]
            row_index_identity += 1
    return check_matrix, pivot_positions

def generate_words_by_sum(generator_matrix):
    num_rows = generator_matrix.shape[0]
    num_cols = generator_matrix.shape[1]
    all_words = np.array([])

    row_indices = list(range(num_rows))
    subsets_of_rows = generate_subsets(row_indices)

    for subset in subsets_of_rows:
        code_word = np.zeros(num_cols)
        if subset:
            for row_index in subset:
                code_word += generator_matrix[row_index]
        code_word %= 2
        all_words = np.append(all_words, code_word)

    all_words = np.resize(all_words, (2 ** num_rows, num_cols))
    all_words = all_words.astype(int)
    all_words = np.unique(all_words, axis=0)
    return all_words

def compute_distance(matrix):
    num_rows = matrix.shape[0]
    num_cols = matrix.shape[1]
    min_distance = num_cols
    for i in range(num_rows - 1):
        for j in range(i + 1, num_rows):
            xor_result = sum((matrix[i] + matrix[j]) % 2)
            if xor_result < min_distance:
                min_distance = xor_result
    return min_distance, min_distance - 1

class LinearCode:
    def __init__(self, matrix):
        self.generator_matrix = matrix
        self.transformed_matrix = row_echelon_form(matrix)
        self.check_matrix, self.pivots = create_check_matrix(self.transformed_matrix)

    def get_shape(self):
        num_rows = self.transformed_matrix.shape[0]
        num_cols = self.transformed_matrix.shape[1]
        return num_cols, num_rows

def validate_code(generator_matrix, check_matrix):
    words_sum = generate_words_by_sum(generator_matrix)
    print('\n Все кодовые слова по сумме G:', words_sum, sep='\n')

    multi_result = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0],
                             [0, 0, 1, 1, 1], [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 1, 0], [0, 1, 0, 1, 1],
                             [0, 1, 1, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1], [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 1], [1, 0, 0, 1, 0], [1, 0, 0, 1, 1], [1, 0, 1, 0, 0], [1, 0, 1, 0, 1],
                             [1, 0, 1, 1, 0], [1, 0, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 1, 0, 1, 0],
                             [1, 1, 0, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    words_multi = multi_result @ generator_matrix % 2
    words_multi = np.unique(words_multi, axis=0)
    print(' ')
    print('Все кодовые слова c умножением на G:', words_multi, sep='\n')
    print('Массивы одинаковы' if np.array_equal(words_multi, words_sum) else 'Массивы разные')

    check_result = words_multi @ check_matrix % 2
    print(' ')
    print('Умножение кодовых слов на проверочную матрицу:', check_result, sep='\n')

primary_array = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                          [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                          [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                          [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                          [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]])

print('Матрица в ступенчатом виде:')
print(row_echelon_form(primary_array), '\n')
print('Матрица в приведённом виде:')
print(reduced_row_echelon_form(primary_array), '\n')

code = LinearCode(primary_array)
gen_matrix = code.transformed_matrix
lead_positions = code.pivots
check_matrix_result = code.check_matrix

print('G:', gen_matrix, sep='\n')
print('Результат: ', code.get_shape(), '\n')
print('lead: ', lead_positions)
print('H:', check_matrix_result, sep='\n')

validate_code(gen_matrix, check_matrix_result)

print('\nd = ', compute_distance(gen_matrix)[0])
print('t = ', compute_distance(gen_matrix)[1])

vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
error_vector_1 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
error_vector_2 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1])

result_1 = (vector + error_vector_1) @ check_matrix_result % 2
print(result_1, '- ошибка' if np.any(result_1) else '- без ошибок')

result_2 = (vector + error_vector_2) @ check_matrix_result % 2
print(result_2, '- ошибка' if np.any(result_2) else '- без ошибок')