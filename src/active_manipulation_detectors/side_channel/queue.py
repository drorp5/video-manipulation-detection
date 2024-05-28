from collections import deque


class FixedSizeNaryQueue:
    """N-ary queue of fixed size"""

    def __init__(self, base: int, max_size: int):
        self.queue = deque(maxlen=max_size)
        self.max_size = max_size
        self.base = base
        self.stored_value = 0
        self.length = 0

    def enqueue(self, item: int) -> None:
        if item >= self.base:
            raise ValueError(f"Only values less than {self.base} are allowed as items.")
        if self.is_full():
            self.dequeue()
        self.queue.append(item)
        self.stored_value = self.base * self.stored_value + item
        self.length += 1

    def dequeue(self) -> int:
        try:
            prev_len = self.length
            digit = self.queue.popleft()
            self.stored_value -= digit * self.base ** (prev_len - 1)
            self.length -= 1
            return digit
        except IndexError:
            raise IndexError("Dequeue from an empty queue")

    def is_full(self) -> bool:
        return len(self) == self.max_size

    def __len__(self) -> int:
        return self.length

    def __repr__(self):
        return f"FixedSizeNaryQueue({list(self.queue)})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, FixedSizeNaryQueue):
            return NotImplemented
        return self.stored_value == other.stored_value

    def __getitem__(self, ind: int) -> int:
        return self.queue[ind]

    def peak_first(self) -> int:
        return self.queue[0]


if __name__ == "__main__":
    # Example usage
    fifo = FixedSizeNaryQueue(base=10, max_size=3)
    fifo.enqueue(1)
    fifo.enqueue(2)
    fifo.enqueue(3)
    print(fifo)  # Output: FixedSizeQueue([1, 2, 3])
    print(fifo.stored_value)
    print(fifo[-1])

    fifo.enqueue(4)
    print(fifo)  # Output: FixedSizeQueue([2, 3, 4])

    item = fifo.dequeue()
    print(item)  # Output: 2
    print(fifo)  # Output: FixedSizeQueue([3, 4])
    print(fifo.stored_value)
