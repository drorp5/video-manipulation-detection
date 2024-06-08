from active_manipulation_detectors import ValidationStatus
from active_manipulation_detectors.side_channel.validation import (
    DataValidatorKSymbolsDelayed,
)


class DataValidatorKSymbolsDelayedChanged(DataValidatorKSymbolsDelayed):
    def _validate(self) -> ValidationStatus:
        if len(self.transmitted_data) < self.search_window:
            return ValidationStatus.Incomplete

        last_used_transmitted_index = 0
        for symbol in self.received_data:
            try:
                matched_index = self.transmitted_data[last_used_transmitted_index:last_used_transmitted_index+self.search_window].index(symbol)
                last_used_transmitted_index += max(matched_index, 1)
            except ValueError:
                return ValidationStatus.Invalid
        return ValidationStatus.Valid