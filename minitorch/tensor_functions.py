"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the negation of the input tensor.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            t1: The input tensor.

        Returns:
        -------
            The negation of t1.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            The local derivative with respect to the input.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the inverse of the input tensor.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            t1: The input tensor.

        Returns:
        -------
            The inverse of t1.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            The local derivative with respect to the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the addition of two input tensors.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            t1: The first input tensor.
            t2: The second input tensor.

        Returns:
        -------
            The sum of t1 and t2.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the local derivative with respect to both inputs.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            A tuple containing the local derivatives with respect to both inputs.

        """
        return grad_output, grad_output


class All(Function):
    """All function: returns 1 if all elements are true, 0 otherwise"""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Compute whether all elements are true along a given dimension.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The input tensor.
            dim: The dimension along which to check.

        Returns:
        -------
            A tensor with 1 if all elements are true, 0 otherwise.

        """
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the multiplication of two input tensors.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            t1: The first input tensor.
            t2: The second input tensor.

        Returns:
        -------
            The product of t1 and t2.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the local derivative with respect to both inputs.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            A tuple containing the local derivatives with respect to both inputs.

        """
        (t1, t2) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, t2), grad_output.f.mul_zip(
            grad_output, t1
        )


class Sigmoid(Function):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the sigmoid of the input tensor.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            t1: The input tensor.

        Returns:
        -------
            The sigmoid of t1.

        """
        ctx.save_for_backward(t1)
        return t1.f.sigmoid_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            The local derivative with respect to the input.

        """
        (t1,) = ctx.saved_values
        sigmoid_t1 = t1.f.sigmoid_map(t1)
        sigmoid_t1_neg = sigmoid_t1.f.neg_map(sigmoid_t1)
        sigmoid_t1_neg_sqr = sigmoid_t1_neg.f.mul_zip(sigmoid_t1_neg, sigmoid_t1)
        return grad_output.f.mul_zip(
            grad_output, sigmoid_t1.f.add_zip(sigmoid_t1, sigmoid_t1_neg_sqr)
        )  # $f'(x) = f(x) * (1 - f(x))$


class ReLU(Function):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the ReLU of the input tensor.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            t1: The input tensor.

        Returns:
        -------
            The ReLU of t1.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            The local derivative with respect to the input.

        """
        (t1,) = ctx.saved_values
        return t1.f.relu_back_zip(t1, grad_output)


class Log(Function):
    """Natural logarithm function $f(x) = ln(x)$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the natural logarithm of the input tensor.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            t1: The input tensor.

        Returns:
        -------
            The natural logarithm of t1.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            The local derivative with respect to the input.

        """
        (t1,) = ctx.saved_values
        return t1.f.log_back_zip(t1, grad_output)


class Exp(Function):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the exponential of the input tensor.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            t1: The input tensor.

        Returns:
        -------
            The exponential of t1.

        """
        ctx.save_for_backward(t1)
        return t1.f.exp_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            The local derivative with respect to the input.

        """
        (t1,) = ctx.saved_values
        exp_t1 = t1.f.exp_map(t1)
        return grad_output.f.mul_zip(grad_output, exp_t1)


class Sum(Function):
    """Sum function: computes the sum of elements along a given dimension"""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Compute the sum of elements along a given dimension.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The input tensor.
            dim: The dimension along which to sum. If None, sum over all dimensions.

        Returns:
        -------
            The sum of elements along the specified dimension.

        """
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            A tuple containing the local derivative with respect to the input and a placeholder float.

        """
        return grad_output, 0.0


class LT(Function):
    """Less than function $f(x, y) = 1 if x < y else 0$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the element-wise less than comparison of two input tensors.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            t1: The first input tensor.
            t2: The second input tensor.

        Returns:
        -------
            A tensor with 1 where t1 < t2, and 0 otherwise.

        """
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the local derivative with respect to both inputs.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            A tuple containing zero tensors as the local derivatives.

        """
        return zeros(grad_output.shape), zeros(grad_output.shape)


class EQ(Function):
    """Equality function $f(x, y) = 1 if x == y else 0$"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the element-wise equality comparison of two input tensors.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            t1: The first input tensor.
            t2: The second input tensor.

        Returns:
        -------
            A tensor with 1 where t1 == t2, and 0 otherwise.

        """
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the local derivative with respect to both inputs.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            A tuple containing zero tensors as the local derivatives.

        """
        return zeros(grad_output.shape), zeros(grad_output.shape)


class IsClose(Function):
    """IsClose function: checks if two tensors are element-wise close"""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute whether two tensors are element-wise close.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            t1: The first input tensor.
            t2: The second input tensor.

        Returns:
        -------
            A tensor with 1 where t1 is close to t2, and 0 otherwise.

        """
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    """Permute function: permutes the dimensions of a tensor"""

    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Permute the dimensions of the input tensor.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The input tensor.
            order: The new order of dimensions.

        Returns:
        -------
            The permuted tensor.

        """
        ctx.save_for_backward(order)
        order_list = []
        for i in range(order.size):
            order_list.append(int(order[i]))
        return a._new(a._tensor.permute(*order_list))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            A tuple containing the local derivative with respect to the input and a placeholder float.

        """
        (order,) = ctx.saved_values
        inv_order = [0] * order.size
        for i in range(order.size):
            inv_order[int(order[i])] = i
        return grad_output._new(grad_output._tensor.permute(*inv_order)), 0.0


class View(Function):
    """Change the shape of a tensor"""

    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Change the shape of the input tensor.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The input tensor.
            shape: The new shape of the tensor.

        Returns:
        -------
            The reshaped tensor.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference gradient.

    Args:
    ----
        f: The function to differentiate.
        vals: The input tensors.
        arg: The argument to differentiate with respect to.
        epsilon: The perturbation size.
        ind: The index to perturb.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
