import pathlib
import time
import cv2
import pandas as pd # pandas 필요 (results.pandas() 사용 시)

#  Make sure you have pandas installed: pip install pandas
#  Library import - adjust if using ultralytics
from yolov5 import YOLOv5 # or from ultralytics import YOLO

# 🔧 Workaround for PosixPath error
import platform
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
try:
    # model = YOLO("best.pt") # Ultralytics example
    model = YOLOv5("best.pt", device="cpu")
    print("YOLOv5 model loaded successfully.")
    # --- Get class names (important for drawing labels) ---
    # This depends on your specific yolov5 library implementation.
    # For ultralytics YOLO object: model.names
    # For the yolov5 library you are using, it might be different.
    # Let's assume it's accessible via a 'names' attribute for now.
    # If this causes an error, you'll need to find how your library stores class names.
    try:
        class_names = model.names # Common attribute name
        print(f"Class names loaded: {class_names}")
    except AttributeError:
        print("Warning: Could not automatically get class names from model. Labels might be missing.")
        # You might need to define them manually if detection provides only class IDs:
        # class_names = ['person', 'car', ...] # Example
        class_names = None # Set to None if names cannot be retrieved

except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    exit(1)

# Variables for FPS calculation
prev_time = time.time()
fps = 0

# Prediction interval settings
prediction_interval = 3
count = 0
last_detections = [] # Store the latest detection results (boxes, scores, labels)

# --- Drawing Settings ---
box_color = (0, 255, 0) # Green color for boxes
text_color = (255, 255, 255) # White color for text
box_thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6

print("Starting real-time feed with detection overlays updated periodically (press 'q' to quit)")

while True:
    current_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame.")
        break

    # Always start with the fresh frame for display
    display_frame = frame.copy()

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
                    label = f"{class_names[class_id]}" if class_names and 0 <= class_id < len(class_names) else f"Class {class_id}"

                    current_detections.append({
                        'box': (xmin, ymin, xmax, ymax),
                        'conf': conf,
                        'label': label
                    })
                last_detections = current_detections # Update the stored detections

            except AttributeError as e_pandas:
                 print(f"Could not use 'results.pandas()'. Error: {e_pandas}")
                 print("Attempting fallback or check your YOLO library's result format.")
                 # Add alternative ways to extract data here if needed, e.g., directly from tensors
                 last_detections = [] # Clear detections if extraction fails

            except Exception as e_extract:
                 print(f"Error extracting detection data: {e_extract}")
                 last_detections = [] # Clear detections on error


        except Exception as e_predict:
            print(f"Error during prediction: {e_predict}")
            # Optionally clear last_detections if prediction fails
            # last_detections = []

    # --- Drawing Logic (runs every frame using stored detections) ---
    if last_detections: # Check if there are any detections to draw
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
            label_ymin = max(ymin, text_height + 10) # Ensure text isn't drawn off-screen top
            cv2.rectangle(display_frame, (xmin, label_ymin - text_height - baseline), (xmin + text_width, label_ymin), box_color, -1) # Filled background
            cv2.putText(display_frame, label_text, (xmin, label_ymin - baseline), font, font_scale, text_color, box_thickness)


    # --- FPS Calculation and Display ---
    delta_time = current_time - prev_time
    fps = 1.0 / delta_time if delta_time > 0 else 0
    prev_time = current_time
    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), font, 0.7, (0, 255, 0), 2)

    # --- Update Counter and Show Frame ---
    count += 1
    if count >= prediction_interval:
        count = 0

    cv2.imshow("YOLOv5 Realtime with Periodic Overlay", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit requested.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Program finished.")