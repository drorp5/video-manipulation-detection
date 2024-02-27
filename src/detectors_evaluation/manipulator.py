from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass

class Injector(ABC):
    @abstractmethod
    def inject(self, frame_1: np.ndarray, frame_2: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
    
    
class FullFrameInjector(Injector):
    """Manipulation method is replacing the entire frame with a new one."""
    def __init__(self, fake_img: np.ndarray, dst_shape: Optional[Tuple[int, int]]=None):
        self.fake_img = fake_img
        self.dst_shape = dst_shape
        if dst_shape is not None:
            self.fake_img = cv2.resize(self.fake_img, dst_shape)
            
    def inject(self, frame_1: np.ndarray, frame_2: np.ndarray) -> np.ndarray:
        assert frame_1.shape == frame_2.shape, 'SHAPE ERROR: frames must be same size'
        return cv2.resize(self.fake_img, (frame_1.shape[1], frame_1.shape[0]))
        
    @property
    def name(self):
        return 'full_frame_injector'


class StripeInjector(Injector):
    """Manipulation method is replacing stipe of the upper part of the frame."""
    def __init__(self, fake_img: np.ndarray, first_row: int, last_row: int, dst_shape: Optional[Tuple[int, int]]=None):
        self.fake_img = fake_img
        self.first_row = first_row
        self.last_row = last_row
        if dst_shape is not None:
            self.fake_img, self.first_row , self.last_row = self._resize(self.fake_img, dst_width=dst_shape[0], dst_height=dst_shape[1])
            
    def _resize(self, img: np.ndarray, dst_width: int, dst_height: int) -> Tuple[np.ndarray, int, int]:
        if img.shape[1] == dst_width and img.shape[1] == dst_height:
            return img, self.first_row, self.last_row
        resized = cv2.resize(img, (dst_width, dst_height))
        resize_factor = resized.shape[0] / img.shape[0]
        first_row = int(np.floor(self.first_row * resize_factor))
        last_row = int(np.ceil(self.last_row * resize_factor))
        return resized, first_row, last_row
        
    def inject(self, frame_1: np.ndarray, frame_2: np.ndarray) -> np.ndarray:
        assert frame_1.shape == frame_2.shape, 'SHAPE ERROR: frames must be same size'
        if self.fake_img.shape != frame_1.shape:
            fake_img, first_row, last_row = self._resize(self.fake_img, dst_width=frame_1.shape[1], dst_height=frame_1.shape[0])
        else:
            fake_img = self.fake_img
            first_row = self.first_row
            last_row = self.last_row
        fake_frame = frame_2.copy()
        fake_frame[:last_row-first_row+1, :, :] = fake_img[first_row:last_row+1, :, :]
        return fake_frame

    @property
    def name(self):
        return 'stripe_injector'

@dataclass
class RectangularBoundaries:
    top_row: int
    bottom_row: int
    left_col: int
    right_col: int

class SignPatchInjector(Injector):
    """Manipulation method is inserting patch of sign in upper part of the second frame and fill the stripe with the previous one."""
    def __init__(self, sign_img: np.ndarray, side_length:int, first_row: int=0, last_row: Optional[int]=None):
        assert sign_img.shape[0]==sign_img.shape[1], 'sign image must be square'
        self.sign_img = sign_img
        self.side_length = side_length
        self.first_row = first_row
        if last_row is None:
            self.last_row = self.sign_img.shape[0]-1
        else:
            self.last_row = last_row
        self.center = (self.sign_img.shape[0]//2, self.sign_img.shape[0]//2)
        self.octagon_mask = self.create_octagon_mask()
        self.num_cols = self.octagon_mask.shape[1]
        self.num_rows = self.last_row - self.first_row + 1
        
    def create_octagon_mask(self) -> np.ndarray:
        mask = np.zeros_like(self.sign_img)
        octagon_color = 255
        angle_offset = np.pi / 8
        vertices = []
        for i in range(8):
            angle = i * np.pi / 4 + angle_offset
            x = int(self.center[0] + self.side_length * np.cos(angle))
            y = int(self.center[1] + self.side_length * np.sin(angle))
            vertices.append((x, y))
        cv2.fillPoly(mask, [np.array(vertices)], (octagon_color, octagon_color, octagon_color))
        return mask
    
    def _get_injection_boundaries(self, frame:np.ndarray, destination_center=None) -> RectangularBoundaries:
        if destination_center is None:
            destination_center = (self.num_rows//2, frame.shape[1]-self.sign_img.shape[1]//2)
            
        top_row = destination_center[0] - self.num_rows//2
        bottom_row = destination_center[0] + self.num_rows//2 + self.num_rows % 2
        left_col = destination_center[1] - self.num_cols//2
        right_col = destination_center[1] + self.num_cols//2 + self.num_cols % 2
        return RectangularBoundaries(top_row=top_row, bottom_row=bottom_row, left_col=left_col, right_col=right_col)
       
    def _inject_sign_patch(self, frame: np.ndarray, boundaries: RectangularBoundaries) -> np.ndarray:
        dst_frame = frame.copy()
        frame_cropped = frame[boundaries.top_row:boundaries.bottom_row,boundaries.left_col:boundaries.right_col,:]
        dst_frame[boundaries.top_row:boundaries.bottom_row, boundaries.left_col:boundaries.right_col, :] = np.where(
            ~self.octagon_mask[self.first_row:self.last_row+1, :, :], frame_cropped, self.sign_img[self.first_row:self.last_row+1, :, :])
        return dst_frame
        
    def inject(self, frame_1: np.ndarray, frame_2: np.ndarray) -> np.ndarray:
        assert frame_1.shape == frame_2.shape, 'SHAPE ERROR: frames must be same size'
        boundaries = self._get_injection_boundaries(frame_1)
        frame_1_injected = self._inject_sign_patch(frame=frame_1, boundaries=boundaries)
        fake_frame = frame_2.copy()
        fake_frame[:self.num_rows, :, :] = frame_1_injected[boundaries.top_row:boundaries.bottom_row, :, :]
        return fake_frame

    @property
    def name(self):
        return 'sign_patch_injector'
