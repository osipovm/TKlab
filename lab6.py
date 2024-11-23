import numpy as np
import random

def encode(data, generator):
    return np.polymul(data, generator) % 2

def apply_random_bit_flips(bits, positions):
    for pos in positions:
        bits[pos] ^= 1
    return bits

def add_random_errors(bits, count):
    positions = random.sample(range(len(bits)), count)
    print(f"Ошибки в позициях: {positions}")
    return apply_random_bit_flips(bits, positions)

def add_error_block(bits, length):
    size = len(bits)
    start = random.randint(0, size - length)
    end = (start + length - 1) % size
    print(f"Блок ошибок в диапазоне: {start}-{end}")
    for offset in range(length):
        bits[(start + offset) % size] ^= 1
    return bits

def trim_zeros(array):
    return np.trim_zeros(np.trim_zeros(array, 'f'), 'b')

def is_error_detectable(syndrome, limit):
    trimmed = trim_zeros(syndrome)
    return 0 < len(trimmed) <= limit

def correct_errors(encoded, syndrome, generator, limit, check_block):
    length = len(encoded)

    for shift in range(length):
        error_poly = np.zeros(length, dtype=int)
        error_poly[length - shift - 1] = 1
        shifted_syndrome = np.polymul(syndrome, error_poly) % 2
        reduced_syndrome = np.polydiv(shifted_syndrome, generator)[1] % 2

        if check_block and is_error_detectable(reduced_syndrome, limit):
            return apply_correction(encoded, reduced_syndrome, shift, generator)
        elif not check_block and sum(reduced_syndrome) <= limit:
            return apply_correction(encoded, reduced_syndrome, shift, generator)

    return None

def apply_correction(encoded, reduced_syndrome, shift, generator):
    length = len(encoded)
    correction_poly = np.zeros(length, dtype=int)
    correction_poly[shift - 1] = 1
    correction = np.polymul(correction_poly, reduced_syndrome) % 2
    corrected = np.polyadd(correction, encoded) % 2
    return np.array(np.polydiv(corrected, generator)[0] % 2).astype(int)

def decode(encoded, generator, limit, check_block):
    syndrome = np.polydiv(encoded, generator)[1] % 2
    return correct_errors(encoded, syndrome, generator, limit, check_block)

def analyze_code(generator, message, error_function, error_param, error_limit, check_block):
    print(f"Начальное сообщение: {message}")
    encoded = encode(message, generator)
    print(f"Закодировано: {encoded}")
    with_errors = error_function(encoded.copy(), error_param)
    print(f"С ошибками: {with_errors}")
    decoded = decode(with_errors, generator, error_limit, check_block)
    print(f"Раскодировано: {decoded}")
    if np.array_equal(message, decoded):
        print("Результат совпадает.\n")
    else:
        print("Результат не совпадает.\n")

def analyze_7_4():
    print("Анализ кода (7,4)")
    generator = np.array([1, 1, 0, 1])
    message = np.array([1, 0, 1, 0])
    for error_count in range(1, 4):
        analyze_code(generator, message, add_random_errors, error_count, 1, False)

def analyze_15_9():
    print("Анализ кода (15,9)")
    generator = np.array([1, 0, 0, 1, 1, 1, 1])
    message = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0])
    for block_length in range(1, 5):
        analyze_code(generator, message, add_error_block, block_length, 3, True)

if __name__ == '__main__':
    analyze_7_4()
    analyze_15_9()