"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Initialize the history of the tensor."""
        self.history = History()

    def requires_grad(self) -> bool:
        """Check if the tensor has a history variable already."""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns Converted tensor storage to numpy array"""
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create a tensor of zeros with the given shape."""

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the tensor is constant by checking if it has no history."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Return the parents of the tensor."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to the tensor."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Backpropagate the gradient of the tensor."""
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns
        size of the tensor

        """
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns
        number of dimensions of the tensor

        """
        return len(self._tensor.shape)

    # Tensor functions of add, sub, mul, lt, eq, gt, neg
    # and their overloaded operators

    def add(self, other: TensorLike) -> Tensor:
        """Compute the addition of two tensors."""
        return Add.apply(self, self._ensure_tensor(other))

    def sub(self, other: TensorLike) -> Tensor:
        """Compute the subtraction of two tensors."""
        return Add.apply(self, Neg.apply(self._ensure_tensor(other)))

    def mul(self, other: TensorLike) -> Tensor:
        """Compute the multiplication of two tensors."""
        return Mul.apply(self, self._ensure_tensor(other))

    def lt(self, other: TensorLike) -> Tensor:
        """Compute the less than comparison of two tensors."""
        return LT.apply(self, self._ensure_tensor(other))

    def eq(self, other: TensorLike) -> Tensor:
        """Compute the equality comparison of two tensors."""
        return EQ.apply(self, self._ensure_tensor(other))

    def gt(self, other: TensorLike) -> Tensor:
        """Compute the greater than comparison of two tensors."""
        return LT.apply(self._ensure_tensor(other), self)

    def neg(self) -> Tensor:
        """Compute the negation of a tensor."""
        return Neg.apply(self)

    def __add__(self, other: TensorLike) -> Tensor:
        return self.add(other)

    def __radd__(self, other: TensorLike) -> Tensor:
        return self.add(other)

    def __sub__(self, other: TensorLike) -> Tensor:
        return self.sub(other)

    def __rsub__(self, other: TensorLike) -> Tensor:
        return self._ensure_tensor(other).sub(self)

    def __mul__(self, other: TensorLike) -> Tensor:
        return self.mul(other)

    def __rmul__(self, other: TensorLike) -> Tensor:
        return self._ensure_tensor(other).mul(self)

    def __lt__(self, other: TensorLike) -> Tensor:
        return self.lt(other)

    def __eq__(self, other: TensorLike) -> Tensor:  # type: ignore
        return self.eq(other)

    def __gt__(self, other: TensorLike) -> Tensor:
        return self.gt(other)

    def __neg__(self) -> Tensor:
        return self.neg()

    # Tensor functions of sigmoid, relu, log, exp, sum, mean

    def sigmoid(self) -> Tensor:
        """Compute the sigmoid function of a tensor."""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Compute the ReLU function of a tensor."""
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Compute the natural logarithm of a tensor."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Compute the exponential of a tensor."""
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Compute the sum of a tensor."""
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean of a tensor."""
        sum_result = self.sum(dim)
        if dim is None:  # If no dimension is specified, use the total size
            return sum_result.mul(1.0 / self.size)
        else:  # If a dimension is specified, use the size of that dimension
            return sum_result.mul(1.0 / self.shape[dim])

    # Tensor functions of all, is_close, permute, view

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Compute the logical AND of all elements in a tensor."""
        if (
            dim is None
        ):  # If no dimension is specified, use the total size and change the shape
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:  # If a dimension is specified, use the size of that dimension only
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, other: TensorLike) -> Tensor:
        """Compute the element-wise close comparison of two tensors."""
        return IsClose.apply(self, self._ensure_tensor(other))

    def permute(self, *order: int) -> Tensor:
        """Permute the dimensions of a tensor."""
        return Permute.apply(self, tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        """Change the shape of a tensor."""
        return View.apply(self, tensor(list(shape)))

    # Special ensor functions of zero_grad_

    def zero_grad_(self) -> None:
        """Reset the gradient of a tensor."""
        self.grad = None
