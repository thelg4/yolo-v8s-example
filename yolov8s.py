import cv2
import numpy as np
import torch
from ultralytics import YOLO

def run_yolo_inference(video_path=None, source_cam=0, save_output=True, conf_threshold=0.25):
    """
    Run YOLOv8s inference on video frames
    
    Args:
        video_path: Path to video file (None for webcam)
        source_cam: Camera device index (default 0)
        save_output: Whether to save the output video
        conf_threshold: Confidence threshold for detections
    
    Returns:
        None (displays video with detections)
    """
    # Load YOLOv8s model
    model = YOLO('yolov8s.pt')
    
    # Initialize video source
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(source_cam)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if saving output
    if save_output:
        output_path = 'yolo_inference_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break
            
        frame_count += 1
        
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=conf_threshold)
        
        # Visualize results on the frame
        annotated_frame = results[0].plot()
        
        # Display frame with detections
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        # Save frame if required
        if save_output:
            out.write(annotated_frame)
        
        # Use a shorter wait time for real-time processing
        # but make sure we can still exit with 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            print("User terminated the process.")
            break
            
        # Safety check - if no frames processed in 5 seconds, exit
        if video_path is None and frame_count > 150:  # Assuming ~30fps, check after 5 seconds
            if not ret:
                print("No frames being received from camera. Exiting.")
                break
    
    # Clean up
    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Inference completed. Output saved to 'yolo_inference_output.mp4'" if save_output else "Inference completed.")

def process_single_image(image_path, conf_threshold=0.25, save_output=True):
    """
    Process a single image with YOLOv8s
    
    Args:
        image_path: Path to image file
        conf_threshold: Confidence threshold for detections
        save_output: Whether to save the output image
    """
    # Load YOLOv8s model
    model = YOLO('yolov8s.pt')
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Run inference
    results = model(image, conf=conf_threshold)
    
    # Print detection results
    print(f"Detections in {image_path}:")
    for i, detection in enumerate(results[0].boxes):
        class_id = int(detection.cls)
        class_name = results[0].names[class_id]
        confidence = float(detection.conf)
        bbox = detection.xyxy[0].tolist()  # x1, y1, x2, y2 format
        print(f"  {i+1}. {class_name}: {confidence:.2f} at position {[round(x) for x in bbox]}")
    
    # Visualize results
    annotated_image = results[0].plot()
    
    # Display result
    cv2.imshow("YOLOv8 Inference", annotated_image)
    print("Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save output if required
    if save_output:
        output_path = 'yolo_output.jpg'
        cv2.imwrite(output_path, annotated_image)
        print(f"Output saved to {output_path}")
        
    # Return detection results for potential further use
    return results[0]

def batch_process_frames(frames_directory, conf_threshold=0.25, save_output=True):
    """
    Process a directory of frames with YOLOv8s
    
    Args:
        frames_directory: Directory containing frame images
        conf_threshold: Confidence threshold for detections
        save_output: Whether to save the output images
    """
    import os
    import glob
    
    # Load YOLOv8s model
    model = YOLO('yolov8s.pt')
    
    # Create output directory if saving results
    if save_output:
        output_dir = os.path.join(frames_directory, 'yolo_output')
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in directory
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(frames_directory, f'*.{ext}')))
    
    # Process each image
    for i, image_path in enumerate(sorted(image_files)):
        print(f"Processing frame {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            continue
        
        # Run inference
        results = model(image, conf=conf_threshold)
        
        # Visualize results
        annotated_image = results[0].plot()
        
        # Save output if required
        if save_output:
            base_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"yolo_{base_name}")
            cv2.imwrite(output_path, annotated_image)
    
    if save_output:
        print(f"All frames processed. Results saved to {output_dir}")
    else:
        print("All frames processed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8s Inference on Frames or Video')
    parser.add_argument('--mode', type=str, default='video', choices=['video', 'image', 'batch'],
                        help='Inference mode: video, single image, or batch processing')
    parser.add_argument('--source', type=str, default=None,
                        help='Source path (video file, image path, or directory)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (for video mode without source)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save output')
    
    args = parser.parse_args()
    
    # Run inference based on mode
    if args.mode == 'video':
        run_yolo_inference(
            video_path=args.source,
            source_cam=args.camera,
            save_output=not args.no_save,
            conf_threshold=args.conf
        )
    elif args.mode == 'image':
        if not args.source:
            print("Error: Image path is required for image mode")
        else:
            process_single_image(
                image_path=args.source,
                conf_threshold=args.conf,
                save_output=not args.no_save
            )
    elif args.mode == 'batch':
        if not args.source:
            print("Error: Directory path is required for batch mode")
        else:
            batch_process_frames(
                frames_directory=args.source,
                conf_threshold=args.conf,
                save_output=not args.no_save
            )