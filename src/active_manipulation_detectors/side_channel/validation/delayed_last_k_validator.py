from active_manipulation_detectors.side_channel.validation import DataValidator
from active_manipulation_detectors.side_channel.queue import FixedSizeNaryQueue
from active_manipulation_detectors.validation_status import ValidationStatus


class DataValidatorKSymbolsDelayed(DataValidator):
    def __init__(
        self, num_symbols: int, symbols_for_detection: int, max_delay: int = 0
    ) -> None:
        super().__init__(num_symbols=num_symbols, queue_size=symbols_for_detection)
        self.symbols_for_detection = symbols_for_detection
        self.max_delay = max_delay
        self.search_window = max_delay + symbols_for_detection
        self.detected_delay = None

        self.delayed_received_queues = [
            FixedSizeNaryQueue(base=num_symbols, max_size=symbols_for_detection)
            for _ in range(max_delay)
        ]

        self.all_received_queues = [self.received_data]
        self.all_received_queues.extend(self.delayed_received_queues)

    def add_received_data(self, received_symbol: int) -> None:
        insertion_value = self.received_data.peak_first()
        self.received_data.enqueue(received_symbol)
        for current_queue in self.delayed_received_queues:
            current_first_val = current_queue.peak_first()
            current_queue.enqueue(insertion_value)
            insertion_value = current_first_val

    def _validate(self) -> ValidationStatus:
        if not self.all_received_queues[-1].is_full():
            return ValidationStatus.Incomplete
        for delay, queue in enumerate(self.all_received_queues):
            if queue == self.transmitted_data:
                self.detected_delay = delay
            return ValidationStatus.Valid
        self.detected_delay = None
        return ValidationStatus.Invalid
