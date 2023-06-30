
def multiply_matrix(first, second):

    if len(first[0]) != len(second):
        raise ArithmeticError("Matrix cannot be multiplied")

    result = [[sum(a * b for a, b in zip(row_a, col_b)) for
               col_b in zip(*second)] for row_a in first]

    return result


def transpose_matrix(m):
    result = [[m[-j][-k] for j in range(len(m))] for k in range(len(m[0]))]

    return result
