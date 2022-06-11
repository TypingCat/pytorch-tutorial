import pyrealsense2 as rs
import numpy as np
import cv2
import torch


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
    
    # Load YOLO model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Start streaming
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data()) # Convert images to numpy arrays
            
            # Object detection
            results = model(color_image)           

            # Show images
            results.render()    # Update result image with boxes and labels
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', results.imgs[0])
            if cv2.waitKey(1) & 0xFF == 27:     # Escape when ESC pressed
                break
    finally:
        pipeline.stop()
    