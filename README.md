# TemplateMatcher

## Overview

`TemplateMatcher` is a Python library designed to optimize and simplify template matching operations using OpenCV. It provides high-level, easy-to-use functionality for image processing tasks, particularly in pattern recognition and object detection. The core of the library is the `TemplateMatcher` class, which comes with extensive documentation and comprehensive docstrings to ensure clarity and ease of use.

## Features

-   **Optimized Template Matching**: Streamlines the process of template matching in Python by leveraging various OpenCV methods for efficient image analysis.
-   **High-Level Functionality**: Provides accessible functions for common template matching tasks, suitable for both novice and experienced users.
-   **Resource and Output Management**: Automates the creation of directories for storing resources (such as images) and output data, facilitating organized project management.
-   **Flexible Image Handling**: Generates and uses test images, supports the use of alpha channel masks, and allows for template matching under various conditions.
-   **Comprehensive Results**: Stores results in both visual (images) and data (JSON) formats, providing a thorough analysis of template matching outcomes.
-   **Comprehensive Documentation**: Each method and attribute is well documented with clear docstrings, making the library user-friendly and easy to integrate.

## Methods

-   `__init__`: Initializes the class, creates directories, and prepares test images.
-   `generate_test_images`: Creates and stores test images for quick setup of template matching experiments.
-   `match_template`: Performs template matching on the source image using a specified method. Optionally uses a mask if the template image supports it.
-   `match_template_alpha`: Similar to `match_template` but specifically for templates with an alpha channel.
-   `_match_template`: A private method for advanced users who wish to directly access lower-level template matching functionality.
-   `get_method_name`: Retrieves the string representation of the OpenCV template matching method used.
-   `save_results`: Records template matching results as images and JSON data, aiding in result analysis and presentation.
-   `test`: Facilitates testing with various template matching methods, handling result saving automatically.

## Usage

### Basic Usage

To use `TemplateMatcher` for standard template matching tasks:

```python
from template_matcher import TemplateMatcher
import cv2

# Load your images
source_img = cv2.imread('path/to/source/image.png')
template_img = cv2.imread('path/to/template/image.png')

# Create a TemplateMatcher instance
matcher = TemplateMatcher()

# Perform standard template matching
coordinates = matcher.match_template(source_img, template_img)
```

> To generate new test images, run the `dev_image_generator.py`.

#### Template Match with transparent alpha channel

> If you use the `match_template_alpha` with a image without alpha channel, it will

```python
from template_matcher import TemplateMatcher
import cv2

# Load your images
source_img = cv2.imread('path/to/source/image.png')
template_img = cv2.imread('path/to/template/image.png') # Template with alpha channel

# Create a TemplateMatcher instance
matcher = TemplateMatcher()

# For templates with an alpha channel
alpha_coordinates = matcher.match_template_alpha(source_img, template_img)
```

### Advanced Usage

For more experienced Python users who wish to utilize the private `_match_template` function:

```python
# Using the private `_match_template` method directly
advanced_result = matcher._match_template(source_img, template_img, mask=True, method=cv2.TM_SQDIFF_NORMED)
print(advanced_result.get_coords())
```

### Running Tests

To run a series of tests with different template matching methods:

```shell
python path/to/template_matcher.py
```

This will execute the `test` method in the `TemplateMatcher` class, performing template matching using various methods and saving the results.

## Attributes

-   `_dir` (Path): The path to the directory containing the script.
-   `resources_dir` (Path): Directory where resource images are stored.
-   `output_dir` (Path): Directory for saving output data.
-   `source_path` (Path): Path to the source image for template matching.
-   `template_path` (Path): Path to the template image used for matching.

## Applications

`TemplateMatcher` is ideal for various applications in pattern recognition, object detection, and image processing. Its intuitive design makes it a valuable tool for comparing the effectiveness of different OpenCV template matching methods, catering to both research and practical needs in image analysis.
