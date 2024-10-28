import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# 1. Display Test Images
def list_images(images, cols=2, rows=3, cmap=None):
    plt.figure(figsize=(10, 8))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

# 2. Color Selection using HSL
def HSL_color_selection(image):
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    white_mask = cv2.inRange(converted_image, np.uint8([0, 200, 0]), np.uint8([255, 255, 255]))
    yellow_mask = cv2.inRange(converted_image, np.uint8([10, 0, 100]), np.uint8([40, 255, 255]))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

# 3. Canny Edge Detection
def canny_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

# 4. Region of Interest Selection
def region_selection(image):
    mask = np.zeros_like(image)
    ignore_mask_color = 255
    rows, cols = image.shape[:2]
    vertices = np.array([[
        [cols * 0.1, rows * 0.95],
        [cols * 0.4, rows * 0.6],
        [cols * 0.6, rows * 0.6],
        [cols * 0.9, rows * 0.95]
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask)

# 5. Hough Transform
def hough_transform(image):
    return cv2.HoughLinesP(image, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=300)

# 6. Averaging and Extrapolating Lane Lines
def average_slope_intercept(lines):
    left_lines, left_weights = [], []
    right_lines, right_weights = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1, x2 = int((y1 - intercept) / slope), int((y2 - intercept) / slope)
    return (x1, int(y1)), (x2, int(y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1, y2 = image.shape[0], int(image.shape[0] * 0.6)
    return pixel_points(y1, y2, left_lane), pixel_points(y1, y2, right_lane)

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=10):
    line_image = np.zeros_like(image)
    for line in lines:
        if line:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

# 7. Frame Processing for Video Stream
def frame_processor(image):
    color_select = HSL_color_selection(image)
    edges = canny_detector(color_select)
    region = region_selection(edges)
    lines = hough_transform(region)
    if lines is not None:
        lanes = lane_lines(image, lines)
        return draw_lane_lines(image, lanes)
    return image

# 8. Processing Video
def process_video(input_path, output_path):
    input_video = VideoFileClip(input_path)
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(output_path, audio=False)

# Run the Code
if __name__ == "__main__":
    # Process each video file
    video_files = ["challenge.mp4", "solidWhiteRight.mp4", "solidYellowLeft.mp4"]
    input_folder = "test_videos"  # Folder where your input videos are stored
    output_folder = "output_videos"  # Folder where processed videos will be saved

    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video in video_files:
        input_path = os.path.join(input_folder, video)
        output_path = os.path.join(output_folder, f"output_{video}")
        print(f"Processing {video}...")
        process_video(input_path, output_path)
        print(f"Processed {video} saved as {output_path}")
