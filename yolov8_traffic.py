import cv2
from ultralytics import YOLO

# Load YOLOv8 model (can be 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO('yolov8n.pt')  # Make sure this file is present in the project directory

def detect_objects(video_path='traffic.mp4'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run detection on the frame
        results = model(frame, stream=True)

        # Draw bounding boxes on frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame in byte format for Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

