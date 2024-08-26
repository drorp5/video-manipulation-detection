from abc import ABC, abstractmethod
from typing import Optional
import logging
import threading

from gige.handlers.gige_handler import GigeHandler


class Car(ABC):
    """
    Abstract base class for car implementations.

    This class provides a framework for implementing different types of cars
    with camera functionality and logging capabilities.
    """

    @abstractmethod
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        camera_started_event: Optional[threading.Event] = None,
        camera_stopped_event: Optional[threading.Event] = None,
    ) -> None:
        """
        Initialize the Car.

        Args:
            logger (Optional[logging.Logger]): Logger object for logging messages.
            camera_started_event (Optional[threading.Event]): Event to signal camera start.
            camera_stopped_event (Optional[threading.Event]): Event to signal camera stop.
        """
        super().__init__()
        self.logger = logger
        self.shutdown_event = threading.Event()
        self.camera_started_event = camera_started_event
        self.camera_stopped_event = camera_stopped_event

    @abstractmethod
    def _run(self) -> None:
        """
        Abstract method to be implemented by subclasses.
        This method should contain the main logic for running the car.
        """
        pass

    def run(self) -> None:
        """
        Run the car's main logic until shutdown is requested.
        """
        while not self.shutdown_event.is_set():
            self._run()

    @abstractmethod
    def get_handler(self) -> GigeHandler:
        """
        Abstract method to get the GigE handler for the car.

        Returns:
            GigeHandler: The GigE handler for the car.
        """
        pass

    def log(self, msg, log_level=logging.INFO):
        """
        Log a message using the configured logger or print to console.

        Args:
            msg (str): The message to log.
            log_level (int): The logging level (default: logging.INFO).
        """
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
