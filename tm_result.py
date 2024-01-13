import cv2
import numpy as np


class TMResult:
    """
    Class to hold the results of template matching.

    ### Attributes:
        - image_size (tuple): The size of the image in which template matching was performed.
        - template_size (tuple): The size of the template used for matching.
        - result (np.ndarray): The result of template matching as a NumPy array.

    ### Methods:
        - get_coords(): Returns the top-left coordinates along with the width and height of the matched region.
        - get_center(): Returns the center coordinates of the matched region.
    """
    result = None

    def __init__(self, method, method_name, mask_used, result, image_size, template_size, threshold=1):
        """
        Initializes the TMResult object.

        ### Args
            - method (int): The OpenCV template matching method used.
            - method_name (str): The name of the OpenCV template matching method used.
            - mask_used (bool): Indicates whether a mask was used in template matching.
            - image_size (tuple): Size of the image (width, height) in which template matching was performed.
            - template_size (tuple): Size of the template (width, height) used for matching.
            - result (np.ndarray): Resulting array from template matching.
            - threshold (float): Threshold value for considering a match (default is 1.0).
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
        self._min = (
            None if np.isnan(self._result).all()
            else np.nanmin(self._result)
        )
        self._max = (
            None if np.isnan(self._result).all()
            else np.nanmax(self._result)
        )
        result = self._result.copy()

        match self.method:
            case cv2.TM_SQDIFF_NORMED:
                # Smalles float is the highest match, we will invert this so 1.0 will be a match
                if self.mask_used:
                    # If the min value is smaller than 0, shift all values so the min value is 0
                    if self._min < 0:
                        result -= self._min

                    # set infinity values to nan, since they indicate no match!
                    result[
                        np.isinf(self._result)
                    ] = np.nan

                    # Norm the values between 0 and 1, where 1 is the match now
                    result = np.exp(-result)

                    # Set NaN and empty to 0.0
                    result[
                        np.isnan(result)
                    ] = 0.0
                else:
                    # Invert the results, 1.0 is a full match
                    result = 1 - result
            case (
                cv2.TM_SQDIFF, cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
                cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED
            ):
                # Can be implemented by proerply normalising the results between 0.0 and 1.0
                # You can use differnt methods based on the possible min and max value of the method
                # This is why i wrote the test function in the template matcher, if you did it correct, you
                # will see a match as white and a non match as black.
                pass  # Not Implemented yet!

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
        y_coords, x_coords = np.where(self.result >= self.threshold)
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
            center_x = x + self.template_size[1] // 2
            center_y = y + self.template_size[0] // 2
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
            bottom_right = (
                x + self.template_size[1], y + self.template_size[0])
            result.append([top_left, bottom_right])
        return result

    def get_grayscale_image(self) -> np.ndarray:
        """ Returns a grayscale version of the result, ensuring that the values are clipped between 0 and 255. """
        result = self.result.copy()
        result = np.clip(result, 0, 1)
        return 255 * result
