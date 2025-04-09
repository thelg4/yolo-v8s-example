# YOLOv8s Frame Inference

A simple yet powerful tool for running YOLOv8s object detection on video frames, single images, or batches of frames.

## Features

- Process video files or webcam feeds in real-time
- Analyze single images with detailed object detection
- Batch process directories of image frames
- Configurable confidence threshold
- Option to save or display results

## Installation

1. Clone this repository or download the script
2. Install the required dependencies:

```
pip install requirements.txt
```

The YOLOv8s model weights will be automatically downloaded on first run.

## Usage

### Process a video file:
```
python yolov8-inference.py --mode video --source path/to/video.mp4
```

### Use webcam:
```
python yolov8-inference.py --mode video
```

### Process a single image:
```
python yolov8-inference.py --mode image --source path/to/image.jpg
```

### Batch process frames in a directory:
```
python yolov8-inference.py --mode batch --source path/to/frames_directory
```

## Command Line Arguments

- `--mode`: Inference mode - 'video', 'image', or 'batch' (default: 'video')
- `--source`: Source path for video file, image, or directory (default: None, uses webcam in video mode)
- `--camera`: Camera device index for webcam use (default: 0)
- `--conf`: Confidence threshold for detections (default: 0.25)
- `--no-save`: Flag to disable saving output files (default: False)

## Examples

Run inference on webcam with higher confidence threshold:
```
python yolov8-inference.py --mode video --conf 0.4
```

Process video file without saving the output:
```
python yolov8-inference.py --mode video --source input.mp4 --no-save
```

Batch process all images in a directory with lower confidence threshold:
```
python yolov8-inference.py --mode batch --source ./frames --conf 0.2
```

## Output

- Video mode: Displays detection results in real-time and saves to 'yolo_inference_output.mp4' (unless --no-save is used)
- Image mode: Displays and saves detection results to 'yolo_output.jpg' (unless --no-save is used)
- Batch mode: Saves detection results in a 'yolo_output' subdirectory within the frames directory (unless --no-save is used)

## Functions

The script provides three main functions that can also be imported and used programmatically:

1. `run_yolo_inference()`: For video processing
2. `process_single_image()`: For single image analysis 
3. `batch_process_frames()`: For processing directories of images

## Requirements

- Python 3.6+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
