import cv2
import numpy as np

def add_text_box(img: np.ndarray, text: str) -> np.ndarray:
    dst_img = img.copy()
    # Get text size and desired position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1  # Adjust font size as needed
    text_size, _ = cv2.getTextSize(text, font, font_scale, 1)  # Get text width and height
    text_x = 10  # Adjust horizontal position
    text_y = 100  # Adjust vertical position (TOP of the image)
    # Create a text box with a slightly filled background for better contrast
    text_box_color = (255, 255, 250)  # Adjust background color (white with slight transparency)
    text_thickness = -1  # Fill the text box
    # Draw the filled text box
    cv2.rectangle(dst_img, (text_x - 5, text_y + text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5), text_box_color, text_thickness)

    # Draw the text on top of the text box
    cv2.putText(dst_img, text, (text_x, text_y), font, font_scale, (0, 0, 255), 1)  # Black text
    return dst_img