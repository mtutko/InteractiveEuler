import math

def my_square_root(val):
    """Single line summary of function

    Longer description can do here (or be ignored)

    Note:
        Any note about this function.

    Args:
        val: Value which we will take the square root of

    Returns:
        sqrt(val) if val>0, -1 otherwise.

    TODO:
        List any "to-do" things here. This is useful for collaborating.

    Examples:
        >>> my_square_root(5.0)
        2.23606797749979
        >>> my_square_root(-3.5)
        -1
    """
    if val < 0:
        return -1
    else:
        return math.sqrt(val)