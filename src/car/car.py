from abc import ABC, abstractmethod
from typing import Optional
import logging
import threading

from gige.handlers import GigeHandler


class Car(ABC):
    @abstractmethod
    def __init__(self, logger: Optional[logging.Logger] = None,
                camera_started_event: Optional[threading.Event] = None,
                camera_stopped_event: Optional[threading.Event] = None) -> None:
        super().__init__()
        self.logger = logger
        self.shutdown_event = threading.Event()
        self.camera_started_event = camera_started_event
        self.camera_stopped_event = camera_stopped_event

    @abstractmethod
    def _run(self) -> None:
        pass

    def run(self) -> None:
        while not self.shutdown_event.is_set():
            self._run()

    @abstractmethod
    def get_handler(self) -> GigeHandler:
        pass

    def log(self, msg, log_level=logging.INFO):
        if self.logger is None:
            print(msg)
            return
        if log_level == logging.DEBUG:
            self.logger.debug(msg)
        elif log_level == logging.INFO:
            self.logger.info(msg)
        elif log_level == logging.WARNING:
            self.logger.warning(msg)
        elif log_level == logging.ERROR:
            self.logger.error(msg)
        elif log_level == logging.CRITICAL:
            self.logger.critical(msg)
        else:
            raise ValueError(f"Invalid log level: {log_level}")
