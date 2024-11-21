from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    values1 = list(vals)
    values2 = list(vals)
    values1[arg] += epsilon
    values2[arg] -= epsilon

    f_plus = f(*values1)
    f_minus = f(*values2)
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative on this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique id of this variable."""
        ...

    def is_leaf(self) -> bool:
        """Returns True if this variable is a leaf (created by the user)."""
        ...

    def is_constant(self) -> bool:
        """Returns True if this variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parents of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the chain rule for this variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph."""
    visited = set()
    result = []

    def dfs(var: Variable) -> None:
        """Implement depth first search"""
        if id(var) not in visited and not var.is_constant():
            visited.add(id(var))
            for parent in var.parents:
                dfs(parent)
        else:
            return
        result.append(var)

    dfs(variable)
    return reversed(result)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leave nodes."""
    sorted_vars = topological_sort(variable)
    derivatives = {id(variable): deriv}

    for var in sorted_vars:
        if var.is_leaf():
            var.accumulate_derivative(derivatives[id(var)])
        else:
            for parent, grad in var.chain_rule(derivatives[id(var)]):
                if id(parent) not in derivatives:
                    derivatives[id(parent)] = grad
                else:
                    derivatives[id(parent)] += grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors for the context."""
        return self.saved_values
