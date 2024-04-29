import enum
from typing import Dict, List
import math

from bitarray import bitarray

from bitarray_dict import BitarrayDict


class IntBitsEncoderDecoder:
    def __init__(self, values: List[int]):
        self._bits_per_symbol = math.ceil(math.log2(len(values)))
        self._values_to_symbols = {
            value: self.int_to_bitarray(ind) for ind, value in enumerate(values)
        }
        self._symbols_to_values = BitarrayDict()
        for value, symbol in self._values_to_symbols.items():
            self._symbols_to_values[symbol] = value

    def int_to_bitarray(self, value: int) -> bitarray:
        binary_string = bin(value)[2:]
        padded_binary_string = binary_string.zfill(self.bits_per_symbol)
        return bitarray(padded_binary_string)

    @property
    def bits_per_symbol(self) -> int:
        return self._bits_per_symbol

    @property
    def values_to_symbols(self) -> Dict[int, bitarray]:
        return self._values_to_symbols

    @property
    def symbols_to_values(self) -> BitarrayDict:
        return self._symbols_to_values

    def encode(self, value: int) -> bitarray:
        return self._values_to_symbols[value]

    def decode(self, symbol: bitarray) -> int:
        return self._symbols_to_values[symbol]
