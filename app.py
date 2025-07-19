from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
from yolov8_traffic import detect_objects, detect_image_file
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    videos = [f for f in os.listdir('.') if f.endswith('.mp4')]
    images = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
    return render_template('index.html', videos=videos, images=images)

@app.route('/live/<video>')
def live(video):
    return Response(detect_objects(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected image", 400
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        output_path = detect_image_file(filepath)
        return redirect(url_for('show_image', filename=os.path.basename(output_path)))

@app.route('/detect_image/<filename>')
def show_image(filename):
    return render_template('result.html', filename=filename)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    os.environ["YOLO_CONFIG_DIR"] = "/tmp"
    app.run(host='0.0.0.0', port=5000)


