from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
from gtts import gTTS
import os
from io import BytesIO

app = Flask(__name__)

# Load YOLOv8 model
model_path = r"best.pt"  # Make sure the path is correct
model = YOLO(model_path)

# Object-color mapping (same as before)
object_colors = {
    "APPLE": ((0, 0, 255), "RED"),
    "BANANA": ((0, 255, 255), "YELLOW"),
    "BITTER MELON": ((0, 255, 0), "GREEN"),
    "BROCCOLI": ((0, 255, 0), "GREEN"),
    "CIRCLE": ((0, 255, 0), "GREEN"),
    "CORN": ((0, 255, 255), "YELLOW"),
    "EGGPLANT": ((128, 0, 128), "PURPLE"),
    "GRAPES": ((128, 0, 128), "PURPLE"),
    "MUSHROOM": ((42, 42, 165), "BROWN"),
    "ORANGE": ((0, 165, 255), "ORANGE"),
    "OVAL": ((255, 0, 0), "BLUE"),
    "PEAR": ((0, 255, 255), "YELLOW"),
    "PUMPKIN": ((0, 165, 255), "ORANGE"),
    "RECTANGLE": ((0, 255, 255), "YELLOW"),
    "SQUARE": ((0, 255, 255), "YELLOW"),
    "STAR": ((0, 165, 255), "ORANGE"),
    "STRAWBERRY": ((0, 0, 255), "RED"),
    "TOMATO": ((0, 0, 255), "RED"),
    "TRIANGLE": ((255, 105, 180), "PINK"),
    "WATERMELON": ((0, 255, 0), "GREEN"),
}

# Object detection
cap = cv2.VideoCapture(1)

detected_objects_global = []  # Store detected objects globally

def generate_frames():
    global detected_objects_global
    while True:
        success, frame = cap.read()
        if not success:
            continue

        results = model.predict(frame, conf=0.6)

        detected_objects = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                detected_object = model.names[class_id].upper()
                (bgr_color, color_name) = object_colors.get(detected_object, ((255, 255, 255), "WHITE"))

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
                label = f"{detected_object} ({color_name})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)
                detected_objects.append({"object": detected_object, "color": color_name})

        detected_objects_global = detected_objects  # Update global variable

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detected_objects', methods=['GET'])
def get_detected_objects():
    return jsonify(detected_objects_global)  # Return the global detected objects


@app.route('/speak', methods=['POST'])
def speak():
    print("OK")
    result_string = ", ".join([f"{obj['object']} ({obj['color']})" for obj in detected_objects_global])
    if result_string:
        # Use gTTS to convert the text to speech
        tts = gTTS(text=result_string, lang='en')
        
        # Save the speech to a BytesIO object to avoid saving as a file
        speech_io = BytesIO()
        tts.write_to_fp(speech_io)
        speech_io.seek(0)
        
        # Return the audio file
        return send_file(speech_io, mimetype='audio/mp3', as_attachment=True, download_name="speech.mp3")
    
    return jsonify({"message": "No text provided"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
