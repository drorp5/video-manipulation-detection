from abc import ABC, abstractmethod
from Crypto.Cipher import ARC4
from bitarray import bitarray


class RandomBitsGenerator(ABC):
    def __init__(self, key: bytes, num_bits_per_iteration: int) -> None:
        super().__init__()
        self.key = key
        self._num_bits_per_iteration = num_bits_per_iteration
        self.stored_bits = bitarray()

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> bitarray:
        pass

    @abstractmethod
    def load_bits(self) -> None:
        pass

    @property
    def num_bits_per_iteration(self) -> int:
        return self._num_bits_per_iteration

    @num_bits_per_iteration.setter
    def num_bits_per_iteration(self, num_bits_per_iteration: int):
        self._num_bits_per_iteration = num_bits_per_iteration


class RandomBitsGeneratorRC4(RandomBitsGenerator):
    def __init__(
        self, key: bytes, num_bits_per_iteration: int, initalization_length: int = 1000
    ) -> None:
        super().__init__(key, num_bits_per_iteration)
        self.cipher = ARC4.new(key)
        self.cipher.encrypt(b"\x00" * initalization_length)

    def __iter__(self):
        return self

    def load_bits(self) -> None:
        keystream_byte = self.cipher.encrypt(b"\x00")
        added_bits = bitarray()
        added_bits.frombytes(keystream_byte)
        self.stored_bits.extend(added_bits)

    def __next__(self) -> bitarray:
        if self.stored_bits.count() < self.num_bits_per_iteration:
            self.load_bits()
        res = self.stored_bits[: self.num_bits_per_iteration]
        del self.stored_bits[: self.num_bits_per_iteration]
        return res


if __name__ == "__main__":
    # Example usage:
    key = b"Key"  # Key must be in bytes
    random_bits_generator = RandomBitsGeneratorRC4(
        key, num_bits_per_iteration=4, initalization_length=1000
    )

    # Generate and print the first 100 bits
    num_iterations = 20
    for _ in range(num_iterations):
        bits = next(random_bits_generator)
        print("Generated bit stream:", bits)
