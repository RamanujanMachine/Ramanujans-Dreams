from numba import njit


@njit(cache=True)
def gcd_recursive(a: int, b: int) -> int:
    """
    Computes GCD of a and b
    """
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def get_gcd_of_array(arr) -> int:
    """
    Calculates GCD of a vector.
    Returns 1 immediately if any pair gives 1.
    """
    d = len(arr)
    if d == 0:
        return 0
    result = abs(arr[0])
    for i in range(1, d):
        result = gcd_recursive(result, abs(arr[i]))
        if result == 1:
            return 1
    return result


@njit(cache=True)
def reduce_to_primitive(arr):
    """
    Divide an integer vector by the GCD of its entries so the result is primitive
    (GCD == 1).  No-op when the GCD is 0 (zero vector) or 1.
    :param arr: An integer vector.
    :return: A new vector with the same direction and GCD 1 (or the input unchanged).
    """
    g = get_gcd_of_array(arr)
    if g <= 1:
        return arr.copy()
    return arr // g
