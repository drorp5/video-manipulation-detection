from typing import Any
from bitarray import bitarray


class BitarrayDict(dict):
    def __getitem__(self, key: bitarray) -> Any:
        return super().__getitem__(key.tobytes())

    def __setitem__(self, key: bitarray, value: Any) -> None:
        super().__setitem__(key.tobytes(), value)

    def __delitem__(self, key: bitarray) -> None:
        super().__delitem__(key.tobytes())

    def __contains__(self, key: bitarray) -> bool:
        return super().__contains__(key.tobytes())

    def keys(self):
        return (bitarray(key) for key in super().keys())
