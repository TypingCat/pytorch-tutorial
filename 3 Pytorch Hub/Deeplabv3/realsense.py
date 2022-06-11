import pyrealsense2 as rs
import numpy as np
import cv2
import torch

from PIL import Image
from torchvision import transforms


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
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load Deeplab model
    if not torch.cuda.is_available():
        print("No cuda available")
        exit(0)    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    model.eval()
    model.to('cuda')
    
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # Start streaming
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # Create input tensor
            color_image = np.asanyarray(color_frame.get_data()) # Convert images to numpy arrays
            input_image = Image.fromarray(color_image)
            input_image = input_image.convert("RGB")            
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
            
            # Semantic segmentation
            input_batch = input_batch.to('cuda')
            with torch.no_grad():
                result = model(input_batch)
                output = result['out'][0]
            output_predictions = output.argmax(0)

            # plot the semantic segmentation predictions of 21 classes in each color
            r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
            r.putpalette(colors)
            r = r.convert('RGB')
            semantic_mask = cv2.cvtColor(np.array(r), cv2.COLOR_RGB2BGR)
            
            # Show images
            alpha = 0.5
            blended = cv2.addWeighted(color_image, alpha, semantic_mask, (1-alpha), 0)
            cv2.namedWindow('Deeplab', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Deeplab', blended)
            if cv2.waitKey(1) & 0xFF == 27:     # Escape when ESC pressed
                break
    finally:
        pipeline.stop()
    