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
    result = None

    def __init__(self, method, method_name, mask_used, result, image_size, template_size, threshold=1):
        """
        Initializes the TemplateMatchResult object.

        ### Args
            - method (int): Size of the image (width, height).
            - method_name (str): Size of the image (width, height).
            - image_size (tuple): Size of the image (width, height).
            - template_size (tuple): Size of the template (width, height).
            - result (np.ndarray): Resulting array from template matching.
            - threshold (float): Threshold value for considering a match (default is 0.95).
        """
        self.method: int = method
        self.method_name: str = method_name
        self.mask_used: bool = mask_used
        self.image_size: tuple = image_size
        self.template_size: tuple = template_size
        self._result: np.ndarray = result
        self.threshold: float = threshold

        self.norm_result()

    def norm_result(self) -> None:
        """ Normalizes the result of template matching based on the theoretical range of each method and set it to `self.result`. """
        match self.method:
            case cv2.TM_SQDIFF_NORMED:
                self.method_min = 0
                self.method_max = np.Infinity

                result = np.exp(-self._result)
                # Fix becouse infinity is not 1.0 and replace nan and empty with 0
                result[
                    np.isinf(self._result) & (self._result > 0)
                ] = 1.0  # +infinity -> 1.0
                result[
                    np.isnan(self._result) | (self._result == 0.0)
                ] = 0.0  # NaN & empty -> 0.0

                self._min = (
                    None if np.isnan(self._result).all()
                    else np.nanmin(self._result)
                )
                self._max = (
                    None if np.isnan(self._result).all()
                    else np.nanmax(self._result)
                )
                self.min = (
                    None if np.isnan(result).all()
                    else np.nanmin(result)
                )
                self.max = (
                    None if np.isnan(result).all()
                    else np.nanmax(result)
                )
            case cv2.TM_SQDIFF:
                self.method_min = "?"
                self.method_max = "?"

                result = self._result
                
                self._min = (
                    None if np.isnan(self._result).all()
                    else np.nanmin(self._result)
                )
                self._max = (
                    None if np.isnan(self._result).all()
                    else np.nanmax(self._result)
                )
                self.min = (
                    None if np.isnan(result).all()
                    else np.nanmin(result)
                )
                self.max = (
                    None if np.isnan(result).all()
                    else np.nanmax(result)
                )
            case cv2.TM_CCORR:
                self.method_min = "?"
                self.method_max = "?"

                result = self._result
                
                self._min = (
                    None if np.isnan(self._result).all()
                    else np.nanmin(self._result)
                )
                self._max = (
                    None if np.isnan(self._result).all()
                    else np.nanmax(self._result)
                )
                self.min = (
                    None if np.isnan(result).all()
                    else np.nanmin(result)
                )
                self.max = (
                    None if np.isnan(result).all()
                    else np.nanmax(result)
                )
            case cv2.TM_CCORR_NORMED:
                self.method_min = "?"
                self.method_max = "?"

                result = self._result
                
                self._min = (
                    None if np.isnan(self._result).all()
                    else np.nanmin(self._result)
                )
                self._max = (
                    None if np.isnan(self._result).all()
                    else np.nanmax(self._result)
                )
                self.min = (
                    None if np.isnan(result).all()
                    else np.nanmin(result)
                )
                self.max = (
                    None if np.isnan(result).all()
                    else np.nanmax(result)
                )
            case cv2.TM_CCOEFF:
                self.method_min = "?"
                self.method_max = "?"

                result = self._result
                
                self._min = (
                    None if np.isnan(self._result).all()
                    else np.nanmin(self._result)
                )
                self._max = (
                    None if np.isnan(self._result).all()
                    else np.nanmax(self._result)
                )
                self.min = (
                    None if np.isnan(result).all()
                    else np.nanmin(result)
                )
                self.max = (
                    None if np.isnan(result).all()
                    else np.nanmax(result)
                )
            case cv2.TM_CCOEFF_NORMED:
                self.method_min = "?"
                self.method_max = "?"

                result = self._result
                
                self._min = (
                    None if np.isnan(self._result).all()
                    else np.nanmin(self._result)
                )
                self._max = (
                    None if np.isnan(self._result).all()
                    else np.nanmax(self._result)
                )
                self.min = (
                    None if np.isnan(result).all()
                    else np.nanmin(result)
                )
                self.max = (
                    None if np.isnan(result).all()
                    else np.nanmax(result)
                )
            case _:
                self.method_min = "?"
                self.method_max = "?"

                result = self._result
                
                self._min = (
                    None if np.isnan(self._result).all()
                    else np.nanmin(self._result)
                )
                self._max = (
                    None if np.isnan(self._result).all()
                    else np.nanmax(self._result)
                )
                self.min = (
                    None if np.isnan(result).all()
                    else np.nanmin(result)
                )
                self.max = (
                    None if np.isnan(result).all()
                    else np.nanmax(result)
                )

        self.result = result

    def get_coords(self):
        """
        Calculates and returns the coordinates where the normalized result is greater than the threshold.

        Returns:
            list: A list of tuples containing (x, y) coordinates of matching points.
        """
        y_coords, x_coords = np.where(self.result > self.threshold)
        matching_coords = [(x, y) for x, y in zip(x_coords, y_coords)]
        return matching_coords

    def get_center(self):
        """
        Calculates and returns the center coordinates of the matched template.

        Returns:
            list: A list of tuples containing (x, y) coordinates of the center of the matched regions.
        """
        result = []
        for x, y in self.get_coords():
            center_x = x + self.template_size[0] // 2
            center_y = y + self.template_size[1] // 2
            result.append([center_x, center_y])
        return result

    def get_marker_coords(self):
        """
        Calculates and returns the top-left and bottom-right coordinates of the matched template.

        Returns:
            list: A list of tuples containing two tuples - ((x1, y1), (x2, y2)), 
                   where (x1, y1) are the top-left coordinates and (x2, y2) are the bottom-right coordinates.
        """
        result = []
        for x, y in self.get_coords():
            top_left = (x, y)
            bottom_right = (x + self.template_size[0], y + self.template_size[1])
            result.append([top_left, bottom_right])
        return result

    def get_grayscale_image(self) -> np.ndarray:
        pass
