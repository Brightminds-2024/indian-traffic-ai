from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, jsonify
from yolov8_traffic import detect_objects, detect_image_file
from claude_chat import get_ai_response
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
VIDEO_FOLDER = 'static/videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

@app.route('/')
def index():
    videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]
    images = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
    return render_template('index.html', videos=videos, images=images)

@app.route('/live/<video>')
def live(video):
    video_path = os.path.join(VIDEO_FOLDER, video)
    return Response(detect_objects(video_path),
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

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video file part", 400
    file = request.files['video']
    if file.filename == '':
        return "No selected video", 400
    if file:
        filepath = os.path.join(VIDEO_FOLDER, file.filename)
        file.save(filepath)
        return redirect(url_for('index'))

@app.route('/detect_image/<filename>')
def show_image(filename):
    return render_template('result.html', filename=filename)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/videos/<filename>')
def uploaded_video(filename):
    return send_from_directory(VIDEO_FOLDER, filename)

@app.route('/delete_file', methods=['POST'])
def delete_file():
    filename = request.form['filename']
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(UPLOAD_FOLDER, filename)
    elif filename.endswith('.mp4'):
        path = os.path.join(VIDEO_FOLDER, filename)
    else:
        return "Invalid file", 400

    if os.path.exists(path):
        os.remove(path)
        return redirect(url_for('index'))
    else:
        return "File not found", 404

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    try:
        response_text = get_ai_response(user_input)
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'response': f'⚠️ AI Error: {str(e)}'})

if __name__ == '__main__':
    os.environ["YOLO_CONFIG_DIR"] = "/tmp"
    app.run(host='0.0.0.0', port=5000)

