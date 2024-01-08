
# TemplateMatcher

## Overview
`TemplateMatcher` is a Python class designed for performing template matching using various OpenCV methods. The class facilitates template matching on images and includes functions to generate test images, perform template matching using different methods (with or without a mask), and save the results.

## Features
- Creation of necessary directories for resources and output data.
- Generation of test images for template matching.
- Support for various OpenCV template matching methods.
- Option to use an alpha channel mask for template matching.
- Saving of template matching results as images and JSON data.

## Methods
- `__init__`: Initializes the class, creates directories, and generates test images.
- `generate_test_images`: Creates test images for template matching and saves them in the resources directory.
- `match_template`: Performs template matching on the source image using a specified method. Optionally uses a mask if the template image supports it.
- `get_method_name`: Retrieves the string name corresponding to an OpenCV template matching method.
- `save_results`: Saves the template matching results as images and JSON data in the output directory.
- `test`: Runs template matching tests using various methods and saves the results.

## Usage
Instantiate the class and call the `test` method to perform template matching using different methods and save the results.

```python
from template_matcher import TemplateMatcher

matcher = TemplateMatcher()
matcher.test()
```

## Attributes
- `_dir` (Path): The directory path of the script.
- `resources_dir` (Path): Path to the directory where resource images are stored.
- `output_dir` (Path): Path to the directory where output data will be saved.
- `source_path` (Path): Path to the source image used for template matching.
- `template_path` (Path): Path to the template image used for matching.

## Applications
The `TemplateMatcher` class can be used in applications such as pattern recognition, object detection, and image processing to compare the effectiveness of different template matching methods in OpenCV.
