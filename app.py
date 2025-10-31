import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
import threading

from voice_backend import (
    start_voice_mode, stop_conversation, initialize_agent_and_tools,
    index_new_pdf, get_pdf_names, get_file_hash, pdf_hashes, set_ws_callback
)

UPLOAD_FOLDER = "pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

agent, pdf_names = initialize_agent_and_tools()
voice_thread = None

def ws_push(msg, event='new_message'):
    socketio.emit(event, msg)

set_ws_callback(ws_push)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    filename = secure_filename(file.filename)
    pdf_hash = get_file_hash(file.stream)
    if pdf_hash in pdf_hashes:
        return "Duplicate PDF content detected!", 409
    pdf_hashes.add(pdf_hash)
    file.stream.seek(0)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    index_new_pdf(filepath)
    return "Uploaded", 200

@app.route('/pdfs', methods=['GET'])
def list_pdfs():
    return jsonify({"pdfs": get_pdf_names()})

#@app.route('/pdfs/<filename>')
#def serve_pdf(filename):
#    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/start_talking', methods=['POST'])
def start_talking():
    global voice_thread
    if voice_thread is None or not voice_thread.is_alive():
        voice_thread = threading.Thread(target=start_voice_mode, daemon=True)
        voice_thread.start()
    return 'OK', 200

@app.route('/stop', methods=['POST'])
def stop():
    stop_conversation()
    # Send a "Stopped" system message to the frontend
    ws_push({
        "sender": "System",
        "message": "<b>Stopped:</b> Voice assistant stopped and is not listening.",
        "side": "left"
    }, event='new_message')
    return '', 204

@socketio.on('connect')
def handle_connect():
    print("SocketIO client connected")

if __name__ == '__main__':
    socketio.run(app, port=5001, debug=False)
