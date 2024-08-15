from abc import ABC, abstractmethod
from typing import Any, Union

from .data_holder import DataHolder
from active_manipulation_detectors.side_channel.data_structure.data_holder import (
    DataHolder,
)
from active_manipulation_detectors.side_channel.data_structure.list_holder import (
    ListDataHolder,
)
from active_manipulation_detectors.side_channel.data_structure.bitarray_holder import (
    BitArrayDataHolder,
)


class DataHolderFactory(ABC):
    """
    Factory class for creating DataHolder objects.
    """

    @staticmethod
    def create(data_type: str, data_unit: int = 1) -> DataHolder:
        """
        Creates a DataHolder object based on the specified data type and data unit.

        Args:
            data_type: The type of data the DataHolder should hold (e.g., "list", "bitarray").
            data_unit: The size of the data unit (default 1).

        Returns:
            A DataHolder object of the specified type.

        Raises:
            ValueError: If the data type is not supported.
        """
        if data_type == "list":
            return ListDataHolder(data_unit)
        elif data_type == "bitarray":
            return BitArrayDataHolder(data_unit)
        else:
            raise ValueError("Unsupported data type: {}".format(data_type))
