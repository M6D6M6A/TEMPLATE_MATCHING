import cv2
import numpy as np


class TMResult:
    """
    Class to hold the results of template matching.

    Attributes:
        image_size (tuple): The size of the image in which template matching was performed.
        template_size (tuple): The size of the template used for matching.
        result (np.ndarray): The result of template matching as a NumPy array.

    Methods:
        get_coords(): Returns the top-left coordinates along with the width and height of the matched region.
        get_center(): Returns the center coordinates of the matched region.
    """
    normed_result = None

    def __init__(self, method, method_name, result, image_size, template_size):
        """
        Initializes the TemplateMatchResult object.

        ### Args
            - method (int): Size of the image (width, height).
            - method_name (str): Size of the image (width, height).
            - image_size (tuple): Size of the image (width, height).
            - template_size (tuple): Size of the template (width, height).
            - result (np.ndarray): Resulting array from template matching.
        """
        self.method: int = method
        self.method_name: str = method_name
        self.image_size: tuple = image_size
        self.template_size: tuple = template_size
        self.result: np.ndarray = result

        self.norm_result()

    def norm_result(self) -> None:
        """ Normalizes the result of template matching based on the theoretical range of each method and set it to `self.normed_result`. """
        match self.method:
            case cv2.TM_SQDIFF_NORMED:
                """ min = 0, max = +infinity """
                result = 1 - np.exp(-self.result)
            case cv2.TM_SQDIFF:
                """ min = ?, max = ? """
                result = self.result
            case cv2.TM_CCORR:
                """ min = ?, max = ? """
                result = self.result
            case cv2.TM_CCORR_NORMED:
                """ min = ?, max = ? """
                result = self.result
            case cv2.TM_CCOEFF:
                """ min = ?, max = ? """
                result = self.result
            case cv2.TM_CCOEFF_NORMED:
                """ min = ?, max = ? """
                result = self.result

        self.normed_result = result

    def get_coords(self):
        """
        Calculates and returns the top-left point coordinates and dimensions of the matched template.

        Returns:
            tuple: (x, y, width, height) of the matched region.
        """
        result = self.normed_result
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        bottom_right = (top_left[0] + self.template_size[0],
                        top_left[1] + self.template_size[1])
        return top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]

    def get_center(self):
        """
        Calculates and returns the center coordinates of the matched template.

        Returns:
            tuple: (x, y) coordinates of the center of the matched region.
        """
        x, y, width, height = self.get_coords()
        center_x = x + width // 2
        center_y = y + height // 2
        return center_x, center_y

    def get_marker_coords(self):
        """
        Calculates and returns the top-left and bottom-right coordinates of the matched template.

        Returns:
            tuple: A tuple containing two tuples - ((x1, y1), (x2, y2)), 
                   where (x1, y1) are the top-left coordinates and (x2, y2) are the bottom-right coordinates.
        """
        x, y, width, height = self.get_coords()
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        return top_left, bottom_right
