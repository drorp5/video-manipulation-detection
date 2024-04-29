from enum import IntEnum


class ValidationStatus(IntEnum):
    """
    Data validation status
        Valid  -  data is valid
        Invalid - data is invalid
        Incomplete   - not enough data to compare
    """

    Valid = 0
    Invalid = 1
    Incomplete = 2

    def __str__(self):
        return self._name_
