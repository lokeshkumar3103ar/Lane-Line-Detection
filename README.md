# Lane Detection System

A computer vision project for detecting lane lines in road videos. It uses color selection, edge detection, and Hough transforms to overlay lane lines on input videos.

## Project Structure
- **`lanelines.py`**: Main script for lane detection.
- **`test_videos/`**: Sample input videos (`challenge.mp4`, `solidWhiteRight.mp4`, `solidYellowLeft.mp4`).
- **`output_videos/`**: Folder where processed videos will be saved.

## Features
1. Detects yellow and white lane lines using HSL color filtering.
2. Applies Canny edge detection and region selection for precise lane detection.
3. Uses Hough transforms to draw lane lines on the video output.

## Requirements
Install dependencies:
```bash
pip install opencv-python numpy matplotlib moviepy
```

## Usage
1. Place input videos in `test_videos`.
2. Run the script:
   ```bash
   python lanelines.py
   ```
3. Processed videos will be saved in `output_videos`.

## Code Overview
- **Key Functions**:
  - `HSL_color_selection`: Isolates lane colors.
  - `canny_detector`: Applies edge detection.
  - `region_selection`: Focuses on lanes.
  - `hough_transform` & `draw_lane_lines`: Detects and overlays lane lines.

Happy coding!


