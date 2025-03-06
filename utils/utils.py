import argparse
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from models.model import Model
from matplotlib.animation import FuncAnimation

model = Model()


def argument_parser_video():
    """Arguments parser for video"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, 
                        default='./data/default_video.mp4',
                        help='path to your input video')
    return parser.parse_args()


def argument_parser_image():
    """Argument parser for image"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str,
                        default='./data/default_image.jpg',
                        help='path to your image')
    return parser.parse_args()


def process_video(input_video_path: str, output_path: str):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    print('FPS:', fps)
    
    success, frame = cap.read()
    
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    success = True


    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(frame)  # First frame for initialization

    def update(frame_num):
        success, frame = cap.read()
        if not success:
            return im,

        # Processing frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(frame)
        label = prediction['label']
        conf = prediction['confidence']
        
        # adding text with label on frame
        frame = cv2.putText(frame, label.title(), 
                            (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0, 255, 0), 2, cv2.LINE_AA)

        # Writing the frame to the video
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        im.set_array(frame)  # Updating the image on the plot
        return im,

    # Create animation
    ani = FuncAnimation(fig, update, frames=range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), interval=1000/fps)

    # show animation
    plt.show()

    cap.release()
    out.release()

def process_image(input_image_path: str, output_path: str):
    """Image processing function"""
    image = cv2.imread(input_image_path)
    
    prediction = model.predict(image=image)
    label = prediction['label']
    
    # Show image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(label)
    plt.axis('off')
    plt.show()
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)


    image = cv2.putText(image, label.title(), 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 255, 0), 2, cv2.LINE_AA)

    # Save image
    cv2.imwrite(output_path, image)

