"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable


# ## Task 0.1

# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(num_1: float, num_2: float) -> float:
    """Multiplies two float numbers.

    Args:
    ----
        num_1: The first float number.
        num_2: The second float number.

    Returns:
    -------
        The product result of num_1 and num_2.

    """
    return num_1 * num_2


def id(num: float) -> float:
    """Returns the input float number unchanged.

    Args:
    ----
        num: The input float number.

    Returns:
    -------
        The same value as the input float number.

    """
    return num


def add(num_1: float, num_2: float) -> float:
    """Adds two float numbers.

    Args:
    ----
        num_1: The first float number.
        num_2: The second float number.

    Returns:
    -------
        The sum result of num_1 and num_2.

    """
    return num_1 + num_2


def neg(num: float) -> float:
    """Returns the negation of the input float number.

    Args:
    ----
        num: The input float number.

    Returns:
    -------
        The negation of the input float number.

    """
    return -num


def lt(num_1: float, num_2: float) -> float:
    """Checks if one float number is less than another float number.

    Args:
    ----
        num_1: The first float number.
        num_2: The second float number.

    Returns:
    -------
        The return float value. 1.0 if num_1 is less than num_2, 0.0 otherwise.

    """
    return 1.0 if num_1 < num_2 else 0.0


def eq(num_1: float, num_2: float) -> float:
    """Checks if two float numbers are equal.

    Args:
    ----
        num_1: The first float number.
        num_2: The second float number.

    Returns:
    -------
        The return float value. 1.0 if num_1 is equal to num_2, 0.0 otherwise.

    """
    return 1.0 if num_1 == num_2 else 0.0


def max(num_1: float, num_2: float) -> float:
    """Returns the larger of two float numbers.

    Args:
    ----
        num_1: The first float number.
        num_2: The second float number.

    Returns:
    -------
        The larger of num_1 and num_2.

    """
    if num_1 > num_2:
        return num_1
    else:
        return num_2


def is_close(num_1: float, num_2: float) -> bool:
    """Checks if two float numbers are close in value.

    Args:
    ----
        num_1: The first float number.
        num_2: The second float number.

    Returns:
    -------
        The return value. True if the absolute difference between num_1 and num_2 is less than 1e-2, False otherwise.

    """
    return (num_1 - num_2 < 1e-2) and (
        num_2 - num_1 < 1e-2
    )  # 1e-2 is the tolerance for close numbers ($f(x) = |x - y| < 1e-2$)


def sigmoid(num: float) -> float:
    """Calculates the sigmoid function of the input float number.

    Args:
    ----
        num: The input float number.

    Returns:
    -------
        The sigmoid result of the input float number.

    """
    if num >= 0:
        return 1.0 / (1.0 + math.exp(-num))  # $\frac{1.0}{(1.0 + e^{-x})}$
    else:
        return math.exp(num) / (1.0 + math.exp(num))  # $\frac{e^x}{(1.0 + e^{x})}$


def relu(num: float) -> float:
    """Applies the ReLU activation function to the input float number.

    Args:
    ----
        num: The input float number.

    Returns:
    -------
        The ReLU result of the input float number.

    """
    return num if num > 0 else 0.0


# Define a constant for the tolerance of close numbers
EPS = 1e-6


def log(num: float) -> float:
    """Calculates the natural logarithm of the input float number.

    Args:
    ----
        num: The input float number.

    Returns:
    -------
        The natural logarithm result of the input float number.

    """
    return math.log(num + EPS)


def exp(num: float) -> float:
    """Calculates the exponential function of the input float number.

    Args:
    ----
        num: The input float number.

    Returns:
    -------
        The exponential result of the input float number.

    """
    return math.exp(num)


def inv(num: float) -> float:
    """Calculates the reciprocal of the input float number.

    Args:
    ----
        num: The input float number.

    Returns:
    -------
        The result of 1 divided by the input float number.

    """
    return 1.0 / num


def log_back(num_1: float, num_2: float) -> float:
    """Computes the derivative of log (the first float number) times the second float number.

    Args:
    ----
        num_1: The first float number.
        num_2: The second float number.

    Returns:
    -------
        The derivative of log (the first float number) times the second float number.

    """
    return num_2 / (num_1 + EPS)


def inv_back(num_1: float, num_2: float) -> float:
    """Computes the derivative of reciprocal (the first float number) times the second float number.

    Args:
    ----
        num_1: The first float number.
        num_2: The second float number.

    Returns:
    -------
        The derivative of reciprocal (the first float number) times the second float number.

    """
    return -num_2 / (num_1 * num_1)


def relu_back(num_1: float, num_2: float) -> float:
    """Computes the derivative of ReLU (the first float number) times the second float number.

    Args:
    ----
        num_1: The first float number.
        num_2: The second float number.

    Returns:
    -------
        The derivative of ReLU (the first float number) times the second float number.

    """
    if num_1 > 0:
        return num_2
    else:
        return 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Creates a function that applies a given function to each element of an iterable.

    Args:
    ----
        fn: A function that takes a float number and returns a float number.

    Returns:
    -------
        A function that takes an iterable of float numbers and returns a new iterable
        containing the results of applying fn to each element.

    """

    def _map(lst: Iterable[float]) -> Iterable[float]:
        # Declare a new list to store the results
        ret = []

        # Apply the function to each element in the input iterable
        for x in lst:
            ret.append(fn(x))

        # Return the result
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Creates a function that combines elements from two iterables using a given function.

    Args:
    ----
        fn: A function that takes two float numbers and returns a float number.

    Returns:
    -------
        A function that takes two iterables of float numbers and returns a new iterable
        containing the results of applying fn to pairs of elements from the input iterables.

    """

    def _zipWith(lst_1: Iterable[float], lst_2: Iterable[float]) -> Iterable[float]:
        # Declare a new list to store the results
        ret = []

        # Apply the function to each pair of elements from lst_1 and lst_2
        for x, y in zip(lst_1, lst_2):
            ret.append(fn(x, y))

        # Return the result
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], initial_value: float
) -> Callable[[Iterable[float]], float]:
    """Creates a function that reduces an iterable to a single float number using a given function.

    Args:
    ----
        fn: A function that takes two float numbers and returns a float number.
        initial_value: The initial float number for the reduction.

    Returns:
    -------
        A function that takes an iterable of float numbers and returns a single float value
        resulting from the reduction of the iterable.

    """

    def _reduce(lst: Iterable[float]) -> float:
        # Initialize the result with the initial value
        result = initial_value

        # Apply the function to each element in the input iterable
        for item in lst:
            result = fn(result, item)

        # Return the final result
        return result

    return _reduce


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negate all float numbers in a list using map().

    Args:
    ----
        lst: An iterable of float numbers.

    Returns:
    -------
        An iterable containing the negated float numbers of the input iterable.

    """
    return map(neg)(lst)


def addLists(lst_1: Iterable[float], lst_2: Iterable[float]) -> Iterable[float]:
    """Add corresponding float numbers from two lists using zipWith().

    Args:
    ----
        lst_1: The first iterable of float numbers.
        lst_2: The second iterable of float numbers.

    Returns:
    -------
        An iterable containing the sum of corresponding float numbers from lst_1 and lst_2.

    """
    return zipWith(add)(lst_1, lst_2)


def sum(lst: Iterable[float]) -> float:
    """Sum all float numbers in a list using reduce().

    Args:
    ----
        lst: An iterable of float numbers.

    Returns:
    -------
        The sum of all float numbers in the input iterable.

    """
    return reduce(add, 0.0)(lst)


def prod(lst: Iterable[float]) -> float:
    """Calculate the product of all float numbers in a list using reduce().

    Args:
    ----
        lst: An iterable of float numbers.

    Returns:
    -------
        The product of all float numbers in the input iterable.

    """
    return reduce(mul, 1.0)(lst)
