#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import sys
import os
import json
from picamera2 import Picamera2
from yolov5 import YOLOv5
from libcamera import controls

# --- Get Recording Duration ---
while True:
    try:
        record_duration_str = input("Enter recording duration in seconds (e.g., 30): ")
        record_duration = float(record_duration_str)
        if record_duration <= 0:
            print("Error: Duration must be a positive number.")
        else:
            break
    except ValueError:
        print("Error: Invalid input. Please enter a number.")
print(f"Will record for approximately {record_duration:.1f} seconds.")

# --- Load YOLOv5 model ---
try:
    model = YOLOv5("best.pt", device="cpu")
    print("Model loaded.")
    try:
        class_names = model.names
        print(f"Class names loaded: {class_names}")
    except AttributeError:
        print("Warning: No class names found in the model.")
        class_names = None
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    sys.exit(1)

# --- Initialize Camera using Picamera2 ---
print("Initializing camera with Picamera2...")
picam2 = Picamera2()

awb_enable_setting = True
awb_mode_setting = controls.AwbModeEnum.Auto
print(f"Camera AWB Mode: {'Auto' if awb_enable_setting else 'Manual Fixed'} ({awb_mode_setting if not awb_enable_setting else 'N/A'})")

capture_width = 640
capture_height = 480
target_fps = 30.0
try:
    config = picam2.create_video_configuration(
        main={"size": (capture_width, capture_height), "format": "BGR888"},
        controls={
            "FrameRate": target_fps,
            "AwbEnable": awb_enable_setting,
            "AwbMode": awb_mode_setting
        }
    )
    picam2.configure(config)
    print(f"Picamera2 configured for {capture_width}x{capture_height} BGR @ requested {target_fps} FPS")
except Exception as e:
    print(f"Error configuring Picamera2: {e}")
    sys.exit(1)

picam2.start()
print("Waiting for camera to stabilize and AWB/AE to settle...")
time.sleep(2.0)
print("Camera initialized.")

# --- Frame dimensions ---
frame_width = capture_width
frame_height = capture_height

# --- FPS calculation (for instantaneous display) ---
prev_time_fps = time.time()

# --- Video Writer Initialization ---
codecs_to_try = ['H264', 'X264', 'MP4V']
output_filename_base = "output"
output_filename_ext = ".mp4"
output_filename = f"{output_filename_base}{output_filename_ext}"

if not os.access('.', os.W_OK):
    print(f"Error: No write permission in the current directory ('{os.getcwd()}').")
    if 'picam2' in locals() and picam2.started: picam2.stop()
    sys.exit(1)

out = None
selected_codec = None
save_fps = target_fps

print(f"\nAttempting to initialize VideoWriter for '{output_filename}' at {save_fps:.1f} FPS.")
print(f"Targeting H.264 encoding. Will try codecs: {codecs_to_try}")

for codec_str in codecs_to_try:
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
        print(f"Attempting codec: {codec_str} (FourCC: {fourcc})")
        out = cv2.VideoWriter(output_filename, fourcc, save_fps, (frame_width, frame_height))
        if out.isOpened():
            print(f"VideoWriter opened successfully using {codec_str} codec.")
            selected_codec = codec_str
            break
        else:
            print(f"Warning: Failed to open VideoWriter with {codec_str} codec.")
            if out: out.release(); out = None
    except Exception as e_codec:
         print(f"Warning: Error initializing VideoWriter with {codec_str}: {e_codec}")
         if out: out.release(); out = None

if out is None or not out.isOpened():
    print(f"Error: Could not open VideoWriter with any H.264/MP4V codec tried.")
    print("Please check if necessary codec libraries (e.g., FFmpeg with H.264 support) are installed and accessible by OpenCV.")
    if 'picam2' in locals() and picam2.started: picam2.stop()
    sys.exit(1)

if selected_codec not in ['H264', 'X264']:
     print(f"Note: Could not use H264/X264. Using fallback codec: {selected_codec}")

print(f"\nRecording video to '{output_filename}' using {selected_codec} codec...")
print(f"Attempting to capture and save at {save_fps:.1f} FPS.")
print("NOTE: Actual processing speed may be lower, loop pacing will be applied.")

# --- Setup for Saving Predicted Frames and Metadata ---
output_image_dir = "predicted_frames_with_boxes"
try:
    os.makedirs(output_image_dir, exist_ok=True)
    print(f"Saving predicted frame images (with boxes) to: '{output_image_dir}/'")
except OSError as e:
    print(f"Error creating directory '{output_image_dir}': {e}. Images will not be saved.")
    output_image_dir = None

all_recorded_metadata = []

# --- Main Recording Loop ---
start_time = time.time()
frame_count = 0
processing_fps = 0
prediction_interval = 3
count = 0
last_detections = []

# --- Drawing Settings ---
box_color = (0, 255, 0)
text_color = (255, 255, 255)
box_thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6

# --- Target Frame Duration for Pacing ---
if save_fps > 0:
    target_frame_duration = 1.0 / save_fps
else:
    target_frame_duration = 0
    print("Warning: save_fps is zero, pacing disabled.")

print("\n" + "="*30)
print("Recording started.")
if target_frame_duration > 0:
    print(f"Target Frame Duration for Pacing: {target_frame_duration:.4f} seconds")
print(f"Press 'q' in the OpenCV window to stop early (target duration: {record_duration:.1f}s).")
print("="*30 + "\n")

try:
    while True:
        loop_start_time = time.time()

        current_wall_time = time.time()
        elapsed_time = current_wall_time - start_time
        if elapsed_time >= record_duration:
            print(f"\nRecording finished: Target duration ({record_duration:.1f}s) reached.")
            break

        try:
            frame_from_camera = picam2.capture_array("main")
            frame = cv2.cvtColor(frame_from_camera, cv2.COLOR_RGB2BGR)
        except Exception as e_cap:
            print(f"\nError capturing or converting frame: {e_cap}. Stopping.")
            break

        if frame is None:
            print("\nError: Failed to grab frame (received None). Stopping.")
            break

        display_frame = frame.copy()
        current_frame_number = frame_count
        
        # MODIFICATION: Prepare image save path variable, will be set if detection occurs
        image_path_for_current_detected_frame = None
        image_filename_for_metadata = None # For metadata JSON

        if count == 0:
            try:
                results = model.predict(frame, size=640)
                detections = results.xyxy[0]
                current_detections = []
                can_save_images = output_image_dir is not None

                if len(detections) > 0:
                    # Determine filename and path if detections found and saving is enabled
                    if can_save_images:
                        image_filename_for_metadata = f"predicted_frame_boxes_{current_frame_number}.png"
                        image_path_for_current_detected_frame = os.path.join(output_image_dir, image_filename_for_metadata)

                    for *box, conf, cls_id in detections:
                        xmin, ymin, xmax, ymax = map(int, box)
                        class_id = int(cls_id)
                        confidence = float(conf)
                        label = f"{class_names[class_id]}" if class_names and 0 <= class_id < len(class_names) else f"Class {class_id}"
                        det_info = {'box': (xmin, ymin, xmax, ymax), 'conf': confidence, 'label': label}
                        current_detections.append(det_info)
                        cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), box_color, box_thickness)
                        label_text = f"{label}: {confidence:.2f}"
                        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, box_thickness)
                        label_ymin = max(ymin, text_height + 10)
                        cv2.rectangle(display_frame, (xmin, label_ymin - text_height - baseline), (xmin + text_width, label_ymin), box_color, -1)
                        cv2.putText(display_frame, label_text, (xmin, label_ymin - baseline), font, font_scale, text_color, box_thickness)
                        all_recorded_metadata.append({
                            'frame_index': current_frame_number,
                            'image_filename': image_filename_for_metadata, # Use the determined filename
                            'class_name': label, 'score': f"{confidence:.2f}"
                        })
                last_detections = current_detections
            except AttributeError as e_attr:
                 print(f"\nWarning: Could not access detection results via '.xyxy'. Error: {e_attr}")
                 last_detections = []
            except Exception as e_extract:
                print(f"\nWarning: Error extracting/drawing detection data: {e_extract}")
                last_detections = []
        else:
            if last_detections:
                for det in last_detections:
                    box = det['box']; conf = det['conf']; label = det['label']
                    xmin, ymin, xmax, ymax = box
                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), box_color, box_thickness)
                    label_text = f"{label}: {conf:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, box_thickness)
                    label_ymin = max(ymin, text_height + 10)
                    cv2.rectangle(display_frame, (xmin, label_ymin - text_height - baseline), (xmin + text_width, label_ymin), box_color, -1)
                    cv2.putText(display_frame, label_text, (xmin, label_ymin - baseline), font, font_scale, text_color, box_thickness)

        count += 1
        if count >= prediction_interval: count = 0

        curr_time_fps = time.time()
        time_diff = curr_time_fps - prev_time_fps
        if time_diff > 0: processing_fps = 1.0 / time_diff
        prev_time_fps = curr_time_fps
        remaining_time = max(0, record_duration - elapsed_time)
        info_text = f"Target: {save_fps:.1f} Inst FPS: {processing_fps:.1f} REC Left: {remaining_time:.1f}s"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        try:
            cv2.imshow("Live Video - Recording... (Press 'q' to Quit)", display_frame)
        except cv2.error as e_imshow:
            print(f"\nWarning: cv2.imshow failed: {e_imshow}. Display might not work.")

        # MODIFICATION: Prepare write_frame (this is what goes into the video)
        if display_frame.shape[1] != frame_width or display_frame.shape[0] != frame_height:
             write_frame = cv2.resize(display_frame, (frame_width, frame_height))
        else:
             write_frame = display_frame
        
        # MODIFICATION: Save write_frame as PNG if an image path was set during detection
        if image_path_for_current_detected_frame:
            try:
                cv2.imwrite(image_path_for_current_detected_frame, write_frame)
                # print(f"Saved detected frame: {image_path_for_current_detected_frame}") # Optional: for debugging
            except Exception as e_imwrite:
                print(f"\nWarning: Error saving image {image_path_for_current_detected_frame}: {e_imwrite}")

        # Write to video
        if out.isOpened():
            out.write(write_frame)
        else:
            print("\nError: VideoWriter is not open. Cannot write frame. Stopping.")
            break
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key != 255 and key != -1:
            key_char = chr(key) if key <= 255 else ''
            if key_char == "q":
                print("\nQuit key pressed. Stopping recording early.")
                break

        loop_end_time = time.time()
        loop_elapsed_time = loop_end_time - loop_start_time
        time_to_wait = target_frame_duration - loop_elapsed_time
        if time_to_wait > 0 and target_frame_duration > 0:
            time.sleep(time_to_wait)

finally:
    print("\nReleasing resources...")
    if 'picam2' in locals() and hasattr(picam2, 'started') and picam2.started:
        picam2.stop()
        print("- Picamera2 stopped.")
    if out and out.isOpened():
        out.release()
        print(f"- VideoWriter closed ('{output_filename}').")
    cv2.destroyAllWindows()
    print("- OpenCV windows closed.")

    end_time = time.time()
    total_time = end_time - start_time if 'start_time' in locals() else 0
    print(f"\nScript ran for: {total_time:.2f} seconds.")
    if total_time > 0 and frame_count > 0:
        avg_fps_actual = frame_count / total_time
        print(f"Recorded {frame_count} frames.")
        print(f"Target save FPS was: {save_fps:.1f} FPS.")
        print(f"Average loop FPS achieved (paced): {avg_fps_actual:.2f} FPS.")
    elif frame_count == 0: print("\nNo frames were recorded.")
    else: print("\nRecording finished.")

    if all_recorded_metadata:
        print("\n--- Saving Prediction Metadata ---")
        metadata_filename_json = "predictions_metadata.json"
        try:
            with open(metadata_filename_json, 'w') as f:
                json.dump(all_recorded_metadata, f, indent=4)
            print(f"- Metadata successfully saved to '{metadata_filename_json}'")
        except Exception as e_json:
            print(f"- Error saving metadata to JSON: {e_json}")
    else:
        print("\nNo prediction metadata was recorded during the session.")

print("Script finished.")
