import numpy as np

def value_transform(value):
    value -= 0.4
    value *= 0.15 * 255.0
    value = int(value)
    value = max(min(value, 255), 0)
    return value

def package_data(results, outputs_order):
    output = np.zeros((16, 16, 3), dtype=np.uint8)
    i = 0
    for y in range(16):
        for x in range(16):
            for c in range(3):
                reordered = outputs_order[i]
                value = value_transform(results[reordered])
                output[y, x, c] = value
                i += 1
    return output
