import cv2
from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt')  # Ensure this file exists

def detect_objects(video_path='traffic.mp4'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌  Error: Cannot open video {video_path}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def detect_image_file(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output_path = os.path.join("static/uploads", "output.jpg")
    cv2.imwrite(output_path, img)
    return output_path

