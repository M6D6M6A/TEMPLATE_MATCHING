import json
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np


class TemplateMatcher:
    """
    A class for performing template matching using various OpenCV methods.

    This class is designed to facilitate the process of template matching on images. It includes functions to generate test images, perform template matching using different methods (with or without a mask), and save the results. The class handles the creation of necessary directories for resources and output data.

    ### Methods
        - __init__
            - Initializes the class, creates directories, and generates test images.
        - generate_test_images
            - Creates test images for template matching and saves them in the resources directory.
        - match_template
            - Performs template matching on the source image using a specified method. It can optionally use a mask if the template image supports it.
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

    This class can be used to compare the effectiveness of different template matching methods in OpenCV, particularly useful in applications like pattern recognition, object detection, and image processing.

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

    def __init__(self) -> None:
        """Initializes the TemplateMatcher class and creates necessary directories and test images."""
        self._dir = Path(__file__).resolve().parent
        self.resources_dir = self._dir / 'resources'
        self.resources_dir.mkdir(exist_ok=True)
        self.output_dir = self._dir / 'output'
        self.output_dir.mkdir(exist_ok=True)
        self.source_path = self.resources_dir / 'source.png'
        self.template_path = self.resources_dir / 'template.png'

        self.generate_test_images()

    def generate_test_images(self) -> None:
        """Generates and saves test images for template matching."""
        img_size = (100, 100)
        plus_size = (50, 50)
        line_width = 2
        background_color = 'black'
        plus_color = 'white'

        target_image = Image.new('RGB', img_size, background_color)
        draw = ImageDraw.Draw(target_image)
        plus_coords = [
            (img_size[0] - plus_size[0], img_size[1] -
             plus_size[1] + plus_size[1]//2 - line_width//2),
            (img_size[0], img_size[1] - plus_size[1] +
             plus_size[1]//2 + line_width//2),
            (img_size[0] - plus_size[0] + plus_size[0] //
             2 - line_width//2, img_size[1] - plus_size[1]),
            (img_size[0] - plus_size[0] + plus_size[0] //
             2 + line_width//2, img_size[1])
        ]
        for coord in plus_coords:
            draw.rectangle(coord, fill=plus_color)
        target_image.save(self.source_path)

        search_image = Image.new('RGBA', plus_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(search_image)
        for coord in plus_coords:
            draw.rectangle(coord, fill=(255, 255, 255, 255))
        search_image.save(self.template_path)

    def match_template(self, mask: bool = False, method: int = cv2.TM_CCOEFF_NORMED) -> np.ndarray:
        """
        Performs template matching on an image using a specified OpenCV method, with an option to apply a mask.

        This function reads the source and template images specified in the class instance, and applies the template matching algorithm based on the provided method. An optional alpha channel mask can be used if the template image supports it.

        ### Args

        mask : bool
            - Indicates whether an alpha channel mask should be used for template matching. 
              If True, the alpha channel of the template image is used as a mask. 
              The template image must have an alpha channel (4th channel).

        method : int 
            - The OpenCV template matching method to be used. Available methods are:
                - 0 | cv2.TM_SQDIFF: Squared difference
                - 1 | cv2.TM_SQDIFF_NORMED: Normalized squared difference (recommended if mask=True)
                - 2 | cv2.TM_CCORR: Cross correlation
                - 3 | cv2.TM_CCORR_NORMED: Normalized cross correlation
                - 4 | cv2.TM_CCOEFF: Correlation coefficient
                - 5 | cv2.TM_CCOEFF_NORMED: Normalized correlation coefficient

        ### Returns
            - np.ndarray
                - The result of the template matching, represented as a single-channel image. 
                  Each pixel denotes how much does the template match the source image at that point.

        ### Note
            - The choice of the best method depends on the specific requirements of the application. 
              In general, TM_SQDIFF_NORMED is recommended when using a mask, as it often provides more accurate results.
        """
        img = cv2.imread(str(self.source_path), cv2.IMREAD_UNCHANGED)
        template = cv2.imread(str(self.template_path), cv2.IMREAD_UNCHANGED)

        if mask:
            if template.shape[-1] == 4:
                mask_channel = cv2.split(template)[3]
            else:
                raise ValueError(
                    "Template image does not have an alpha channel for masking.")
        else:
            mask_channel = None

        if img.shape[-1] == 4:
            img = cv2.merge(cv2.split(img)[:3])
        if template.shape[-1] == 4:
            template = cv2.merge(cv2.split(template)[:3])

        return cv2.matchTemplate(img, template, method, mask=mask_channel)

    def get_method_name(self, method: int) -> str:
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

    def save_results(self, method: int, result: np.ndarray, mask: bool) -> None:
        """
        Saves the result images and JSON data for a specific method.

        ### Args
            - method : int
                - The method used for template matching.
            - result : np.ndarray
                - The result array from template matching.
            - mask : bool
                - Indicates if the mask was used in template matching.
        """
        method_name = self.get_method_name(method)
        method_dir = self.output_dir / method_name
        method_dir.mkdir(exist_ok=True)

        file_prefix = "masked_" if mask else ""
        result_img_path = method_dir / f"{file_prefix}out.png"
        grayscale_img_path = method_dir / f"{file_prefix}grayscale_out.png"
        json_path = method_dir / f"{file_prefix}out.json"

        img = cv2.imread(str(self.source_path), cv2.IMREAD_UNCHANGED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc if method in [
            cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED] else min_loc
        bottom_right = (top_left[0] + img.shape[1], top_left[1] + img.shape[0])

        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imwrite(str(result_img_path), img)

        result_normalized = cv2.normalize(
            result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(str(grayscale_img_path), result_normalized)

        match_value = max_val if method in [
            cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED] else min_val
        grayscale_value = result[top_left[1], top_left[0]]

        if np.isinf(grayscale_value):
            grayscale_value_str = "Infinity"
        elif np.isnan(grayscale_value):
            grayscale_value_str = "NaN"
        else:
            grayscale_value_str = int(grayscale_value)

        with open(json_path, 'w') as f:
            json.dump({
                "method": method_name,
                "mask_used": mask,
                "result_location": (top_left, bottom_right),
                "match_value": match_value,
                "grayscale_value": grayscale_value_str
            }, f, indent=4)

    def test(self) -> None:
        """Tests different methods and saves results."""
        methods = [
            cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED
        ]

        for method in methods:
            for mask in [True, False]:
                result = self.match_template(mask=mask, method=method)
                self.save_results(method, result, mask)
                print(f"Results saved for method: {method}, Mask: {mask}")


if __name__ == "__main__":
    matcher = TemplateMatcher()
    matcher.test()
