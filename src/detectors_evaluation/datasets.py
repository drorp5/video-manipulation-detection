from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import cv2


def to_batches(l: List, batch_size: int) -> List:
    """
    Split a list into batches of a specified size.

    Args:
        l (List): The input list to be split.
        batch_size (int): The size of each batch.

    Returns:
        List: A list of batches, where each batch is a sublist of the input list.
    """
    return [l[offset : offset + batch_size] for offset in range(0, len(l), batch_size)]


def read_frame(frame_path: Path) -> np.ndarray:
    """
    Read an RGB image from the specified path.

    Args:
        frame_path (Path): The path to the image file.

    Returns:
        np.ndarray: The image as a NumPy array in RGB format.
    """
    frame = cv2.imread(frame_path.as_posix())
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


class EvaluationDataset(ABC):
    """
    Abstract base class for evaluation datasets.

    This class defines the interface for datasets used in the evaluation process.
    """

    def __init__(self, name: str):
        """
        Initialize the EvaluationDataset.

        Args:
            name (str): The name of the dataset.
        """
        self._index = 0
        self._name = name

    def __iter__(self):
        """
        Return an iterator for the dataset.

        Returns:
            Iterator: An iterator for the dataset.
        """
        return self


def to_batches(l: List, batch_size: int) -> List:
    """
    Split a list into batches of a specified size.

    Args:
        l (List): The input list to be split.
        batch_size (int): The size of each batch.

    Returns:
        List: A list of batches, where each batch is a sublist of the input list.
    """


def read_frame(frame_path: Path) -> np.ndarray:
    """
    Read an RGB image from the specified path.

    Args:
        frame_path (Path): The path to the image file.

    Returns:
        np.ndarray: The image as a NumPy array in RGB format.
    """


class EvaluationDataset(ABC):
    """
    Abstract base class for evaluation datasets.

    This class defines the interface for datasets used in the evaluation process.
    """

    def __init__(self, name: str):
        """
        Initialize the EvaluationDataset.

        Args:
            name (str): The name of the dataset.
        """

    def __iter__(self):
        """
        Return an iterator for the dataset.

        Returns:
            Iterator: An iterator for the dataset.
        """
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the next item in the dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The next pair of frames in the dataset.

        Raises:
            StopIteration: When there are no more items in the dataset.
        """
        if self._index < len(self):
            val = self[self._index]
            self._index += 1
            return val
        else:
            raise StopIteration

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get an item from the dataset by index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A pair of frames from the dataset.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        ...

    @property
    def name(self) -> str:
        """
        Get the name of the dataset.

        Returns:
            str: The name of the dataset.
        """
        return self._name


class FramesDirectoryDataset(EvaluationDataset):
    """
    A dataset class that reads frames from a directory.
    """

    def __init__(self, frames_dir: Path, name: str, min_frame_id: int = 0):
        """
        Initialize the FramesDirectoryDataset.

        Args:
            frames_dir (Path): The directory containing the frame images.
            name (str): The name of the dataset.
            min_frame_id (int, optional): The minimum frame ID to consider. Defaults to 0.
        """
        super().__init__(name=name)
        self.frames_dir = frames_dir
        self.min_frame_id = min_frame_id
        self.all_frames_path = self.get_all_frames_path()
        self.pairs_path = to_batches(self.all_frames_path, batch_size=2)

    def get_all_frames_path(self) -> List[Path]:
        """
        Get paths of all jpg images in the directory ordered by IDs.

        Returns:
            List[Path]: A list of paths to all frame images.
        """
        all_frames_path = []
        for frame_path in self.frames_dir.glob("*.jpg"):
            id = int(frame_path.stem.split("_")[1])
            if id < self.min_frame_id:
                pass
            else:
                all_frames_path.append(frame_path)
        return sorted(all_frames_path, key=lambda x: int(x.stem.split("_")[1]))

    def __len__(self):
        """
        Get the number of frame pairs in the dataset.

        Returns:
            int: The number of frame pairs.
        """
        return len(self.pairs_path)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a pair of frames by index.

        Args:
            idx (int): The index of the frame pair to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A pair of frames from the dataset.
        """
        return read_frame(self.pairs_path[idx][0]), read_frame(self.pairs_path[idx][1])


class VideoDataset(EvaluationDataset):
    """
    A dataset class that reads frames from a video file.
    """

    def __init__(
        self,
        video_path: Path,
        num_frames: int,
        dst_shape: Optional[Tuple[int, int]] = None,
        flip: bool = False,
        name: Optional[str] = None,
    ):
        """
        Initialize the VideoDataset.

        Args:
            video_path (Path): The path to the video file.
            num_frames (int): The number of frames to extract from the video.
            dst_shape (Optional[Tuple[int, int]], optional): The desired shape of the output frames. Defaults to None.
            flip (bool, optional): Whether to flip the frames vertically. Defaults to False.
            name (Optional[str], optional): The name of the dataset. Defaults to None.
        """
        if name is None:
            name = video_path.stem
        super().__init__(name=name)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path.as_posix())
        self.num_frames = num_frames
        self.frames_offset = self.get_linear_spaced_offsets()
        self.dst_shape = dst_shape
        self.flip = flip

    def get_frame(self, index: int) -> np.ndarray:
        """
        Get a single frame from the video.

        Args:
            index (int): The index of the frame to retrieve.

        Returns:
            np.ndarray: The requested frame as a NumPy array.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if self.flip:
            frame = cv2.flip(frame, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.dst_shape is not None:
            frame = cv2.resize(frame, self.dst_shape)
        return frame

    def get_linear_spaced_offsets(self):
        """
        Get frame offsets in equal steps.

        Returns:
            np.ndarray: An array of frame offsets.
        """
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_offset = np.linspace(0, total_frames - 3, self.num_frames).astype(int)
        return frames_offset

    def release(self) -> None:
        """
        Release the video capture object.
        """
        self.cap.release()

    def __len__(self) -> int:
        """
        Get the number of frame pairs in the dataset.

        Returns:
            int: The number of frame pairs.
        """
        return len(self.frames_offset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a pair of frames by index.

        Args:
            idx (int): The index of the frame pair to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A pair of frames from the dataset.
        """
        return self.get_frame(self.frames_offset[idx]), self.get_frame(
            self.frames_offset[idx] + 1
        )


class VideosDirectoryDataset(EvaluationDataset):
    """
    A dataset class that reads frames from multiple videos in a directory.
    """

    def __init__(
        self,
        videos_dir: Path,
        name: str,
        num_frames: int,
        dst_shape: Optional[Tuple[int, int]] = None,
        flip: bool = False,
    ):
        """
        Initialize the VideosDirectoryDataset.

        Args:
            videos_dir (Path): The directory containing the video files.
            name (str): The name of the dataset.
            num_frames (int): The number of frames to extract from each video.
            dst_shape (Optional[Tuple[int, int]], optional): The desired shape of the output frames. Defaults to None.
            flip (bool, optional): Whether to flip the frames vertically. Defaults to False.
        """
        super().__init__(name=name)
        self.videos_dir = videos_dir
        self.videos_path = list(videos_dir.glob("*"))
        self.num_frames = num_frames
        self.dst_shape = dst_shape
        self.flip = flip
        self._video_index = 0
        self._set_cur_dataset()

    def _set_cur_dataset(self):
        """
        Set the current video dataset.
        """
        if self._video_index < len(self.videos_path):
            self.video_dataset = VideoDataset(
                video_path=self.videos_path[self._video_index],
                num_frames=self.num_frames,
                dst_shape=self.dst_shape,
                flip=self.flip,
            )
        else:
            self.video_dataset = None

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a pair of frames by index.

        Args:
            idx (int): The index of the frame pair to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A pair of frames from the dataset.
        """
        return self.video_dataset.__getitem__(idx)

    def __len__(self) -> int:
        """
        Get the number of videos in the dataset.

        Returns:
            int: The number of videos.
        """
        return len(self.videos_path)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Get the next pair of frames across all videos.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The next pair of frames in the dataset.

        Raises:
            StopIteration: When there are no more frames in any video.
        """
        if self._video_index < len(self):
            try:
                val = self.video_dataset.__next__()
                return val
            except StopIteration:
                self.video_dataset.release()
                self._video_index += 1
                self._set_cur_dataset()
                return self.__next__()
            except Exception as e:
                print(
                    f"Error loading video from {self.videos_path[self._video_index]}: {e}"
                )
                self.video_dataset.release()
                self._video_index += 1
                self._set_cur_dataset()
                return self.__next__()
        else:
            raise StopIteration
