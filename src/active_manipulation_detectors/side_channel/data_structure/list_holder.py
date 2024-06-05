from abc import ABC, abstractmethod
from typing import Any, Iterable, Union

from numpy import isin

from active_manipulation_detectors.side_channel.data_structure.data_holder import (
    DataHolder,
)


class ListDataHolder(DataHolder):
    """
    Concrete implementation of DataHolder using a list.
    """

    def __init__(self, data_unit: int = 1) -> None:
        super().__init__(data_unit)
        self._data = []

    def append(self, item: Any) -> None:
        """
        Appends an item to the list.

        Args:
            item: The item to be appended.
        """

        self._validate_input_length(item)

        if isinstance(item, Iterable):
            self._data.extend(list(item))
        else:
            self._data.append(item)

    def __len__(self) -> int:
        """
        Returns the number of items in the list.

        Returns:
            The length of the list.
        """
        return len(self._data) // self._data_unit

    def __getitem__(self, index: Union[int, slice]) -> Any:
        """
        Gets the item at the specified index or a slice of items.

        Args:
            index: The index of the item to retrieve (int) or a slice object
                    specifying a sub-sequence.

        Returns:
            The item at the specified index (for single element access) or
            a new ListDataHolder containing the sliced sub-sequence.

        Raises:
            IndexError: If the index is out of range for single element access.
            TypeError: If the provided index is not an integer or a slice object.
        """
        if isinstance(index, int):
            if not 0 <= index < len(self):
                raise IndexError("Index out of range")
            return self._get_item(index)
        elif isinstance(index, slice):
            # Implement your slicing logic here using the start, stop, and step values from the slice object
            start, stop, step = index.start, index.stop, index.step
            ret = ListDataHolder(data_unit=self._data_unit)
            ret._data = self._data[start:stop:step]
            return ret
        else:
            raise TypeError("Invalid index type. Must be int or slice")

    def _get_item(self, index: int) -> Any:
        """
        Retrieves the item at the specified index from the list.

        Args:
            index: The index of the item to retrieve.

        Returns:
            The item at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        if self._data_unit == 1:
            return self._data[index]
        return self._data[index * self._data_unit : (index + 1) * self._data_unit]

    def __str__(self) -> str:
        return self._data.__str__()

    def __repr__(self) -> str:
        return self._data.__repr__()
