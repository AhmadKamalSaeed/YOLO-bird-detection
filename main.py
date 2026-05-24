import cv2
from ultralytics import YOLO

# Load the smallest YOLO model (it will download automatically the first time)
model = YOLO("yolov8n.pt")

# Open the video file 
video_path = "/Users/ahmadkamal/Documents/Spoor/pigeon-6093.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a single frame
    success, frame = cap.read()
    
    if not success:
        print("Video finished or failed to load.")
        break

    # Pass the frame to YOLO for detection
    results = model(frame)

    # Tell YOLO to draw the bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Pop open a window to show the result
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Press 'q' on your keyboard to quit the video early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up and close windows when done
cap.release()
cv2.destroyAllWindows()