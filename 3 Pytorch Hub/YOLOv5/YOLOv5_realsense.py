# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py

import pyrealsense2 as rs
import numpy as np
import cv2


if __name__ == '__main__':
    pipeline = rs.pipeline()
    config = rs.config()

    # Check color sensor exists
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    if 'RGB Camera' not in [s.get_info(rs.camera_info.name) for s in device.sensors]:
        print("Color sensor not detected")
        exit(0)
    
    # Set data stream for Intel Realsense D435
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

    # Start streaming
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == 27:     # Escape when ESC pressed
                break

    finally:
        pipeline.stop()
    