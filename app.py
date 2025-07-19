from flask import Flask, render_template, Response
from yolov8_traffic import detect_objects  # Make sure this is correctly defined
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # HTML page with video stream

@app.route('/live')
def live():
    # Pass the video file name to the detect_objects function
    return Response(detect_objects("sample_video-2.mp4"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Optional: set YOLO config path to avoid warning
    os.environ["YOLO_CONFIG_DIR"] = "/tmp"
    
    app.run(host='0.0.0.0', port=5000)

