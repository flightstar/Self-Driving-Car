import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import *
from collections import deque

QUEUE_LENGTH=50

class LaneDetector:
    def __init__(self):
        self.left_lines  = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)

    def process(self, image):
        white_yellow = select_white_yellow(image)
        gray         = convert_gray_scale(white_yellow)
        smooth_gray  = apply_smoothing(gray)
        edges        = detect_edges(smooth_gray)
        regions      = select_region(edges)
        lines        = hough_lines(regions)
        left_line, right_line = lane_lines(image, lines)

        def mean_line(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines)>0:
                # Convert array to tuples
                line = np.mean(lines, axis=0, dtype=np.int32)
                # make sure it's tuples not numpy array for cv2.line to work
                line = tuple(map(tuple, line)) 
            return line

        left_line  = mean_line(left_line,  self.left_lines)
        right_line = mean_line(right_line, self.right_lines)

        return draw_lane_lines(image, (left_line, right_line))

# def process_video(test_video, video_input, video_output):
#    detector = LaneDetector()
#    clip = VideoFileClip(os.path.join(test_videos, video_input))
#    processed = clip.fl_image(detector.process)
#    processed.write_videofile(os.path.join('output_videos', video_output), audio=False)
def process_video(test_video, output_video, video_input, video_output):
    detector = LaneDetector()
    clip = VideoFileClip(os.path.join('test_videos', video_input))
    # Modify a clip as you want using custom filters
    processed = clip.fl_image(detector.process)
    # Returns a copy of the clip with a new default fps and filter the audio
    processed.write_videofile(os.path.join('output_videos', video_output), audio=False)
    


