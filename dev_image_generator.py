import json
from pathlib import Path
from PIL import Image, ImageDraw
import cv2

from template_matcher import TemplateMatcher
from tm_result import TMResult


class DevImageGenerator:
    def __init__(self) -> None:
        """Initializes the TemplateMatcher class and creates necessary directories and test images."""
        self._dir = Path(__file__).resolve().parent
        self.resources_dir = self._dir / 'resources'
        self.resources_dir.mkdir(exist_ok=True)
        self.output_dir = self._dir / 'output'
        self.output_dir.mkdir(exist_ok=True)
        self.source_path = self.resources_dir / 'source.png'
        self.template_path = self.resources_dir / 'template.png'

    def generate_test_images(self):
        """Generate and save test images."""
        # Image dimensions and settings
        img_size = (100, 100)
        plus_size = (50, 50)
        line_width = 2
        background_color = 'black'
        plus_color = 'white'

        # Target image with '+' shape
        target_image = Image.new('RGB', img_size, background_color)
        draw = ImageDraw.Draw(target_image)
        plus_coords = [(img_size[0] - plus_size[0], img_size[1] - plus_size[1] + plus_size[1]//2 - line_width//2),
                       (img_size[0], img_size[1] - plus_size[1] +
                        plus_size[1]//2 + line_width//2),
                       (img_size[0] - plus_size[0] + plus_size[0] //
                        2 - line_width//2, img_size[1] - plus_size[1]),
                       (img_size[0] - plus_size[0] + plus_size[0]//2 + line_width//2, img_size[1])]

        # Add the white +
        draw.rectangle(plus_coords[0] + plus_coords[1], fill=plus_color)
        draw.rectangle(plus_coords[2] + plus_coords[3], fill=plus_color)

        # Change the color of the top right 50x50 pixels to white
        draw.rectangle(((50, 0), (100, 50)), fill=(255, 255, 255, 255))

        target_image.save(self.source_path)

        # Template image with alpha channel '+'
        search_image = Image.new('RGBA', plus_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(search_image)
        plus_coords_alpha = [(0, plus_size[1]//2 - line_width//2),
                             (plus_size[0], plus_size[1]//2 + line_width//2),
                             (plus_size[0]//2 - line_width//2, 0),
                             (plus_size[0]//2 + line_width//2, plus_size[1])]
        draw.rectangle(
            plus_coords_alpha[0] + plus_coords_alpha[1], fill=(255, 255, 255, 255))
        draw.rectangle(
            plus_coords_alpha[2] + plus_coords_alpha[3], fill=(255, 255, 255, 255))
        search_image.save(self.template_path)

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

    def save_results(self, method: int, tm_result: TMResult, mask: bool) -> None:
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
        method_name = tm_result.method_name
        method_dir = self.output_dir / method_name
        method_dir.mkdir(exist_ok=True)

        file_prefix = "masked_" if mask else ""
        result_img_path = method_dir / f"{file_prefix}out.png"
        grayscale_img_path = method_dir / f"{file_prefix}grayscale_out.png"
        json_path = method_dir / f"{file_prefix}out.json"

        img = cv2.imread(str(self.source_path), cv2.IMREAD_UNCHANGED)

        for top_left, bottom_right in tm_result.get_marker_coords():
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.imwrite(str(result_img_path), img)

        grayscale_normalized_img = tm_result.get_grayscale_image()
        cv2.imwrite(str(grayscale_img_path), grayscale_normalized_img)

        _json = {
            "method": tm_result.method_name,
            "mask_used": bool(tm_result.mask_used),
            "result_location": [
                [
                    int(c[0]), int(c[1])
                ] for c in tm_result.get_coords()
            ],
            "min_value": float(tm_result._min) if tm_result._min is not None else None,
            "max_value": float(tm_result._max) if tm_result._max is not None else None,
            "normalized_min_value": float(tm_result.min) if tm_result.min is not None else None,
            "normalized_max_val": float(tm_result.max) if tm_result.max is not None else None
        }

        with open(json_path, 'w') as f:
            json.dump(_json, f, indent=4)

    def test(self) -> None:
        """
        Runs template matching tests using various methods and saves the results.
        """
        # Load the source and template images once
        source_img = cv2.imread(str(self.source_path), cv2.IMREAD_UNCHANGED)
        template_img = cv2.imread(
            str(self.template_path), cv2.IMREAD_UNCHANGED)

        methods = [
            cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED
        ]

        for method in methods:
            for mask in [True, False]:
                tm_result = TemplateMatcher._match_template(
                    source_img, template_img, mask=mask, method=method)
                self.save_results(method, tm_result, mask)
                print(f"Results saved for method: {method}, Mask: {mask}")


if __name__ == "__main__":
    matcher = DevImageGenerator()
    matcher.test()
