#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import sys
import os
from picamera2 import Picamera2 # Import picamera2
import pandas as pd #for results.pandas()
from yolov5 import YOLOv5
#import pathlib


# --- Get Recording Duration ---
while True:
    try:
        record_duration_str = input("Enter recording duration in seconds (e.g., 10): ")
        record_duration = float(record_duration_str)
        if record_duration <= 0:
            print("Error: Duration must be a positive number.")
        else:
            break
    except ValueError:
        print("Error: Invalid input. Please enter a number.")

print(f"Will record for {record_duration:.1f} seconds.")

#Load YOLOv5 model

try:

    # Device needs to be changed
    model = YOLOv5("best.pt", device="cpu")
    print("Model loaded.")

    # GET class names
    # this time assume it is models.names
    try:
        class_names = model.names
        print(f"Class names loaded: {class_names}")
    except AttributeError:
        print("No class names loaded.")
        class_names = None
except Exception as e:
    print(f"Error: {e}")
    exit(1)


# --- Initialize Camera using Picamera2 ---
print("Initializing camera with Picamera2...")
picam2 = Picamera2()
# Configure for video recording. Adjust size if needed.
# Check your camera's supported modes if 640x480 fails here.
# Common options: (1456, 1088) for your IMX296, or try lower like (640, 480)
capture_width = 640
capture_height = 480
try:
    config = picam2.create_video_configuration(main={"size": (capture_width, capture_height), "format": "BGR888"}) # Request BGR format directly
    picam2.configure(config)
    print(f"Picamera2 configured for {capture_width}x{capture_height} BGR")
except Exception as e:
    print(f"Error configuring Picamera2: {e}")
    print("Try different dimensions like 1456x1088 or check camera connection.")
    sys.exit(1)

picam2.start()
# Give camera time to start/adjust
time.sleep(1.0)
print("Camera initialized.")

# --- Frame dimensions for VideoWriter ---
# Use the dimensions we configured
frame_width = capture_width
frame_height = capture_height

# --- Background subtractor ---
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# --- Mode and FPS calculation ---
mode = "normal"
prev_time_fps = time.time()

# --- Video Writer Initialization ---
codecs_to_try = ["XVID", "MJPG", "MP4V"]
output_filename_base = "output"
output_filename_ext = ".avi"

if not os.access('.', os.W_OK):
    print(f"Error: No write permission in the current directory ('{os.getcwd()}').")
    picam2.stop()
    sys.exit(1)

out = None
selected_codec = None
for codec_str in codecs_to_try:
    fourcc = cv2.VideoWriter_fourcc(*codec_str)
    if codec_str == "MP4V":
        output_filename = f"{output_filename_base}.mp4"
    else:
        output_filename = f"{output_filename_base}{output_filename_ext}"
    save_fps = 15.0
    print(f"Attempting to initialize VideoWriter with codec {codec_str} at {save_fps:.1f} FPS...")
    out = cv2.VideoWriter(output_filename, fourcc, save_fps, (frame_width, frame_height))
    if out.isOpened():
        print(f"VideoWriter opened successfully using {codec_str} codec.")
        selected_codec = codec_str
        break
    else:
        print(f"Warning: Failed to open VideoWriter with {codec_str} codec.")
        if out: out.release()
        out = None

if out is None or not out.isOpened():
    print(f"Error: Could not open VideoWriter with any tried codec.")
    picam2.stop()
    sys.exit(1)

print(f"Recording video to '{output_filename}'...")

# --- Main Recording Loop ---
start_time = time.time()
frame_count = 0
processing_fps = 0

# ---- Prediction interval settings
prediction_interval = 3
count = 0
last_detections = [] # Store the latest detection results (boxes, scores, labels)

# ---Drawing Setting----
box_color = (0, 255, 0) # Green
text_color = (255, 255, 255) # White
boxc_thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6



print("\n" + "="*30)
print("Recording started.")
print(f"Press 'q' in the OpenCV window to stop early (target duration: {record_duration:.1f}s).")
print("Mode keys: 'n':normal, 't':threshold, 'e':edge, 'b':bg_sub, 'c':contour")
print("="*30 + "\n")

try:
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= record_duration:
            print(f"\nRecording finished: Target duration ({record_duration:.1f}s) reached.")
            break

        # --- Read Frame using Picamera2 ---
        # request = picam2.capture_request() # Use this for more control if needed
        # frame = request.make_array("main")
        # request.release()
        frame = picam2.capture_array("main") # Simpler way to get numpy array

        if frame is None:
            print("\nError: Failed to grab frame using Picamera2. Stopping.")
            break

        # --- Process Frame (OpenCV part) ---
        display_frame = frame.copy() # We already get BGR, no conversion needed initially

        # --- Detection Logic (runs periodically) ---
        if count == 0:
            try:
                results = model.predict(frame, size=640)

                # --- Extract detection data ---
                # This part heavily depends on the specific library's output format.
                # Using results.pandas().xyxy[0] is common for libraries based on Ultralytics.
                # If your library returns something else (like a list of tuples or a raw tensor),
                # you'll need to adapt this data extraction part.
                try:
                    detections_df = results.pandas().xyxy[0]
                    # Filter by confidence if needed (optional)
                    # detections_df = detections_df[detections_df['confidence'] > 0.5]

                    # Store relevant info for drawing later
                    current_detections = []
                    for index, row in detections_df.iterrows():
                        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                        conf = row['confidence']
                        class_id = int(row['class'])
                        label = f"{class_names[class_id]}" if class_names and 0 <= class_id < len(
                            class_names) else f"Class {class_id}"

                        current_detections.append({
                            'box': (xmin, ymin, xmax, ymax),
                            'conf': conf,
                            'label': label
                        })
                    last_detections = current_detections  # Update the stored detections

                except AttributeError as e_pandas:
                    print(f"Could not use 'results.pandas()'. Error: {e_pandas}")
                    print("Attempting fallback or check your YOLO library's result format.")
                    # Add alternative ways to extract data here if needed, e.g., directly from tensors
                    last_detections = []  # Clear detections if extraction fails

                except Exception as e_extract:
                    print(f"Error extracting detection data: {e_extract}")
                    last_detections = []  # Clear detections on error


            except Exception as e_predict:
                print(f"Error during prediction: {e_predict}")
                # Optionally clear last_detections if prediction fails
                # last_detections = []

            # --- Drawing Logic (runs every frame using stored detections) ---
        if last_detections:  # Check if there are any detections to draw
            for det in last_detections:
                box = det['box']
                conf = det['conf']
                label = det['label']
                xmin, ymin, xmax, ymax = box

                # Draw bounding box
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), box_color, box_thickness)

                # Create label text
                label_text = f"{label}: {conf:.2f}"

                # Put label text above the box
                (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, box_thickness)
                label_ymin = max(ymin, text_height + 10)  # Ensure text isn't drawn off-screen top
                cv2.rectangle(display_frame, (xmin, label_ymin - text_height - baseline),
                              (xmin + text_width, label_ymin), box_color, -1)  # Filled background
                cv2.putText(display_frame, label_text, (xmin, label_ymin - baseline), font, font_scale, text_color,
                            box_thickness)
        # --- Update Counter and Show Frame ---
        count += 1
        if count >= prediction_interval:
            count = 0
        try:
            if mode == "threshold":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                display_frame = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
            elif mode == "edge":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            elif mode == "bg_sub":
                fg_mask = bg_subtractor.apply(frame)
                display_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
            elif mode == "contour":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                     cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)
        except cv2.error as e:
            print(f"\nOpenCV error during processing ({mode} mode): {e}")
            print("Falling back to normal mode for this frame.")
            display_frame = frame.copy()

        # --- Calculate and Display FPS & Info ---
        curr_time_fps = time.time()
        time_diff = curr_time_fps - prev_time_fps
        if time_diff > 0:
            processing_fps = 1.0 / time_diff
        prev_time_fps = curr_time_fps
        remaining_time = max(0, record_duration - elapsed_time)
        info_text = f"FPS: {processing_fps:.1f} Mode: {mode} REC Left: {remaining_time:.1f}s"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- Show Live Video ---
        cv2.imshow("Live Video - Recording...", display_frame)

        # --- Write Frame to File ---
        if display_frame.shape[1] != frame_width or display_frame.shape[0] != frame_height:
             write_frame = cv2.resize(display_frame, (frame_width, frame_height))
        else:
             write_frame = display_frame
        out.write(write_frame)
        frame_count += 1

        # --- Handle Key Presses ---
        key = cv2.waitKey(1) & 0xFF
        if key != 255 and key != -1:
            key_char = chr(key)
            if key_char == "t": mode = "threshold"; print(f"\rMode changed to: {mode}      ", end="")
            elif key_char == "e": mode = "edge"; print(f"\rMode changed to: {mode}        ", end="")
            elif key_char == "b": mode = "bg_sub"; print(f"\rMode changed to: {mode}      ", end="")
            elif key_char == "c": mode = "contour"; print(f"\rMode changed to: {mode}     ", end="")
            elif key_char == "n": mode = "normal"; print(f"\rMode changed to: {mode}      ", end="")
            elif key_char == "q":
                print("\nQuit key pressed. Stopping recording early.")
                break

finally:
    # --- Clean up ---
    print("\nReleasing resources...")
    picam2.stop() # Stop the camera
    print("- Picamera2 stopped.")
    if out and out.isOpened():
        out.release()
        print(f"- VideoWriter closed ('{output_filename}').")
    cv2.destroyAllWindows()
    print("- OpenCV windows closed.")

    end_time = time.time()
    total_time = end_time - start_time
    if total_time > 0 and frame_count > 0:
        avg_fps_actual = frame_count / total_time
        print(f"\nRecorded {frame_count} frames in {total_time:.2f} seconds.")
        print(f"Average processing/saving FPS: {avg_fps_actual:.2f} FPS.")
    elif frame_count == 0:
        print("\nNo frames were recorded.")
    else:
        print("\nRecording finished.")

print("Script finished.")
