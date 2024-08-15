from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Union
import numpy as np

class DataHolder(ABC):
    """
    Abstract interface for a data holder object.
    """

    def __init__(self, data_unit: int = 1) -> None:
        super().__init__()
        self._data_unit = data_unit

    @abstractmethod
    def append(self, item: Any) -> None:
        """
        Appends an item to the data holder.

        Args:
            item: The item to be appended.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of items in the data holder.
        """
        pass

    def __eq__(self, other: object) -> TypeError:
        """
        Compares the data holder with another object for equality.

        Args:
            other: The object to compare with.

        Raises:
            TypeError: If the other object is not a subclass of DataHolder.
        """
        if not isinstance(other, DataHolder):
            raise TypeError("Can only compare DataHolder objects")
        return all(self[i] == other[i] for i in range(len(self)))

    def __getitem__(self, index: Union[int, slice]) -> Any:
        """
        Gets the item at the specified index or a slice of items.

        Args:
            index: The index of the item to retrieve (int) or a slice object
                    specifying a sub-sequence.

        Returns:
            The item at the specified index (for single element access) or
            a new object containing the sliced sub-sequence.

        Raises:
            IndexError: If the index is out of range for single element access.
            TypeError: If the provided index is not an integer or a slice object.
        """
        pass

    @abstractmethod
    def _get_item(self, index: int) -> Any:
        """
        Internal method to retrieve the item at the specified index.

        Args:
            index: The index of the item to retrieve.

        Returns:
            The item at the specified index.
        """
        pass

    def _validate_input_length(self, item: Any) -> None:
        if isinstance(item, (int, bool, np.int32, float)):
            if self._data_unit != 1:
                raise ValueError(
                    f"Invalid value length for appended value: must be of length {self._data_unit}"
                )
        elif isinstance(item, Iterable):
            if len(item) != self._data_unit:
                raise ValueError(
                    f"Invalid value length for appended value: must be of length {self._data_unit}"
                )
        else:
            raise ValueError("Invalid value type for DataHolder: {}".format(type(item)))

    @abstractmethod
    def __str__(self) -> str:
        """
        Gets the string representation.

        Returns:
            The string representation.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def get_combinations(self, k: int) -> Iterable[DataHolder]:
        pass

    @abstractmethod
    def __contains__(self, item: Any) -> bool:
        """Check if item is in."""
        pass

    @abstractmethod
    def index(self, item: Any) -> int:
        """Return the index of the first occurrence of item."""
        pass 
        
    @abstractmethod
    def __iter__(self) -> Iterable:
        """Return an iterator."""
        pass
        