import json
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np

from tm_result import TMResult


class TemplateMatcher:
    """
    A class for performing template matching using various OpenCV methods.

    This class is designed to facilitate the process of template matching on images.
    It includes functions to generate test images, perform template matching using different
    methods (with or without a mask), and save the results. The class handles the creation
    of necessary directories for resources and output data.

    ### Methods
        - __init__
            - Initializes the class, creates directories, and generates test images.
        - generate_test_images
            - Creates test images for template matching and saves them in the resources directory.
        - _match_template
            - Performs template matching on the source image using a specified method.
              It can optionally use a mask if the template image supports it.
        - get_method_name
            - Retrieves the string name corresponding to an OpenCV template matching method.
        - save_results
            - Saves the template matching results as images and JSON data in the output directory.
        - test
            - Runs template matching tests using various methods and saves the results.

    ### Usage
        - Instantiate the class.
        - Call the test method to perform template matching using different methods and to save the results.

    ### Example

    The docstring examples assume that `TemplateMatcher` has been imported::

    >>> from template_matcher import TemplateMatcher

    Code snippets are indicated by three greater-than signs::

    >>> matcher = TemplateMatcher()
    >>> matcher.test()

    This class can be used to compare the effectiveness of different template
    matching methods in OpenCV, particularly useful in applications like pattern
    recognition, object detection, and image processing.

    ### Attributes
        - _dir (Path)
            - The directory path of the script.
        - resources_dir (Path)
            - Path to the directory where resource images are stored.
        - output_dir (Path)
            - Path to the directory where output data will be saved.
        - source_path (Path)
            - Path to the source image used for template matching.
        - template_path (Path)
            - Path to the template image used for matching.
    """

    @staticmethod
    def match_template(img: np.ndarray, template: np.ndarray, threshold: float = 1.0) -> list:
        """
        Performs template matching on an image using a specified OpenCV method, with an option to apply a mask.

        ### Args
        img : np.ndarray
            - The source image is what will be used to search for the template in it.
        template : np.ndarray
            - The template image that will be matched to each position of the source.
        threshold : float
            - The threshold used to find the results, if the image is close to the same
              a min value of .95 or lower can be tested if 1.0 does not find it.
              1.0 should work if its a exact match. (Consider hashing the image with md5
              for faster performance if they match 100% and the ability to compare it with
              as many hashes as you want without speed loss)

        ### Returns
        - List of (x, y) Coordinates
        """
        return TemplateMatcher._match_template(img, template, mask=False, method=cv2.TM_SQDIFF_NORMED, threshold=threshold)

    @staticmethod
    def match_template_alpha(
        img: np.ndarray, template: np.ndarray, threshold: float = 1.0, call_warning=None
    ) -> list:
        """
        Performs template matching on an image using a specified OpenCV method, applying a mask. (If the )

        ### Args
        img : np.ndarray
            - The source image is what will be used to search for the template in it.
        template : np.ndarray
            - The template image that will be matched to each position of the source.
        threshold : float
            - The threshold used to find the results, if the image is close to the same a min 
              value of .95 or lower can be tested if 1.0 does not find it.
              1.0 should work if its a exact match. (Consider hashing the image with md5 for 
              faster performance if they match 100% and the ability to compare it with as many 
              hashes as you want without speed loss)
        call_warning : function
            - For example, pass a logger function like `call_warning = logger.warning` to log 
              the warning message to the logger instead of printing it.

        ### Returns
        - List of (x, y) Coordinates
        """
        # Check if the template has an alpha channel
        if template.shape[-1] != 4:
            tm_warning = "Template has no alpha channel, should use `match_template` instead of `match_template_alpha`, but will still work!"
            if call_warning is None:
                print(tm_warning)
            else:
                call_warning(tm_warning)

        return TemplateMatcher._match_template(img, template, mask=True, method=cv2.TM_SQDIFF_NORMED, threshold=threshold)

    @staticmethod
    def _match_template(
            img: np.ndarray, template: np.ndarray,
            mask: bool = False, method: int = cv2.TM_SQDIFF_NORMED, threshold: float = 1.0
    ) -> TMResult:
        """
        Performs template matching on an image using a specified OpenCV method, with an option to apply a mask.

        ### Args
        img : np.ndarray
            - The source image is what will be used to search for the template in it.
        template : np.ndarray
            - The template image that will be matched to each position of the source.
        mask : bool
            - Indicates whether an alpha channel mask should be used for template matching.
        method : int
            - The OpenCV template matching method to be used.
        threshold : float
            - The threshold used to find the results, if the image is close to the same a min 
              value of .95 or lower can be tested if 1.0 does not find it.
              1.0 should work if its a exact match. (Consider hashing the image with md5 for 
              faster performance if they match 100% and the ability to compare it with as many 
              hashes as you want without speed loss)

        ### Returns
        - TMResult
            - The result of the template matching.

        ### Notes
        - Only `method=cv2.TM_SQDIFF_NORMED` is fully implemented, without a mask the other _NORMED
          should probably work as well, if the values are not normalized between 0 and 1 it will not work
          and you have to implement the normalization yourself by finding the possible min and max values
          and remapping the range to 0.0 - 1.0.
        """
        if mask:
            if template.shape[-1] == 4:
                mask_channel = cv2.split(template)[3]
            else:
                mask = False
                mask_channel = None
        else:
            mask_channel = None

        if img.shape[-1] == 4:
            img = cv2.merge(cv2.split(img)[:3])
        if template.shape[-1] == 4:
            template = cv2.merge(cv2.split(template)[:3])

        result = cv2.matchTemplate(img, template, method, mask=mask_channel)
        tm_result = TMResult(
            method, TemplateMatcher.get_method_name(method),
            mask, result, img.shape[:2], template.shape[:2], threshold
        )
        return tm_result

    @staticmethod
    def get_method_name(method: int) -> str:
        """Retrieves the string name of the specified matching method."""
        method_names = {
            cv2.TM_SQDIFF: "TM_SQDIFF",
            cv2.TM_SQDIFF_NORMED: "TM_SQDIFF_NORMED",
            cv2.TM_CCORR: "TM_CCORR",
            cv2.TM_CCORR_NORMED: "TM_CCORR_NORMED",
            cv2.TM_CCOEFF: "TM_CCOEFF",
            cv2.TM_CCOEFF_NORMED: "TM_CCOEFF_NORMED"
        }
        return method_names.get(method, f"Method_{method}")
