from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
    ----
        index : index tuple of ints
        strides : tensor strides

    Returns:
    -------
        Position in storage

    """
    # Initialize position to 0
    stride_position = 0

    # Iterate through corresponding indices and strides simultaneously
    for index, stride in zip(index, strides):
        # For each dimension, multiply index by stride and add to position
        # This converts multidimensional index to flat position using strides
        stride_position += index * stride

    # Return the calculated flat position in storage
    return stride_position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
    ----
        ordinal: ordinal position to convert
        shape : tensor shape
        out_index : return index corresponding to position

    """
    # Start with the ordinal number we want to convert
    remaining_ordinal = ordinal + 0

    # Iterate through dimensions from right to left (least significant to most significant)
    for dimension in range(len(shape) - 1, -1, -1):
        # Get the size of current dimension
        dimension_size = shape[dimension]

        # Extract index for current dimension using modulo (remainder)
        # Convert to int to ensure integer type
        out_index[dimension] = int(remaining_ordinal % dimension_size)

        # Integer divide to get remaining ordinal for next dimensions
        remaining_ordinal = remaining_ordinal // dimension_size


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
    ----
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
    -------
        None

    """
    # Calculate the dimensions of the big and small tensors and their difference
    big_dim = len(big_shape)
    small_dim = len(shape)
    diff = big_dim - small_dim

    # Iterate through dimensions of the small tensor from right to left
    for i in range(small_dim - 1, -1, -1):
        # For dimensions where shape[i] > 1, copy the corresponding index from big_index
        if shape[i] > 1:
            big_i = i + diff  # Map to corresponding dimension in big tensor
            out_index[i] = big_index[big_i]

        # For dimensions where shape[i] == 1, use 0 for broadcasting
        else:
            out_index[i] = 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape1 : first shape
        shape2 : second shape

    Returns:
    -------
        broadcasted shape

    Raises:
    ------
        IndexingError : if cannot broadcast

    """
    # Declare two indices to iterate through the shapes from the right to the left
    shape1_idx = len(shape1) - 1
    shape2_idx = len(shape2) - 1

    # Create the broadcasted shape by checking each pair of values of the two tensor shapes from the right to the left
    broadcasted_shape = []
    while shape1_idx >= 0 or shape2_idx >= 0:
        # Get the value of the current index of both tensor shapes, if the index is out of bounds, set the value to 1
        shape1_val = shape1[shape1_idx] if shape1_idx >= 0 else 1
        shape2_val = shape2[shape2_idx] if shape2_idx >= 0 else 1

        # Check each pari of values of the two tensor shapes based on the broadcasting rules
        if shape1_val == 1:
            broadcasted_shape.insert(0, shape2_val)
        elif shape2_val == 1:
            broadcasted_shape.insert(0, shape1_val)
        elif shape1_val == shape2_val:
            broadcasted_shape.insert(0, shape1_val)
        else:
            raise IndexingError(f"Cannot broadcast shapes {shape1} and {shape2}")

        # Decrement the indices of both tensor shapes by 1
        shape1_idx -= 1
        shape2_idx -= 1

    # Return the broadcasted shape as a tuple
    return tuple(broadcasted_shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a shape"""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert to cuda"""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns
        -------
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Broadcast two shapes to create a new union shape.

        Args:
        ----
            shape_a : The first shape
            shape_b : The second shape

        Returns:
        -------
            The broadcasted shape

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Convert an index into a position.

        Args:
        ----
            index : The index to convert

        Returns:
        -------
            The converted position

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:  # if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generate all valid indices for the tensor.

        Returns
        -------
            An iterable of all valid indices

        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index"""
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Get the value at the given index from the storage.

        Args:
        ----
            key : The index to get the value from

        Returns:
        -------
            The value at the given index

        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Set the value at the given index in the storage.

        Args:
        ----
            key : The index to set the value at in the storage
            val : The value to set in the storage

        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return core tensor data as a tuple."""
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
        ----
            *order: a permutation of the dimensions

        Returns:
        -------
            New `TensorData` with the same storage and a new dimension order

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # Declare two new lists to store the new shape and strides
        new_shape = []
        new_strides = []

        # Create new shape and strides based on the given order
        for i in order:
            new_shape.append(self.shape[i])
            new_strides.append(self._strides[i])

        # Convert the lists to tuples
        new_shape = tuple(new_shape)
        new_strides = tuple(new_strides)

        # Return a new TensorData object with permuted dimensions
        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        """Convert to string"""
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
