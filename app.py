#!/usr/bin/env python3
import http.server
import json
import os
import socketserver
import subprocess
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PORT = int(os.environ.get('PORT', 8877))
BASE_DIR = Path(__file__).parent

UPLOAD_DIR = Path('/tmp/surreal_uploads')
UPLOAD_DIR.mkdir(exist_ok=True)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

jobs = {}  # job_id -> {process, lines, done, returncode}


def analyze_clip(video_path, out_w, out_h):
    try:
        import cv2
        import math
    except ImportError:
        return {'error': 'opencv not available'}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'cannot open video'}

    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    # Sample 3 frames; keep the one with the best face detection
    best_frame, best_faces = None, []
    for pos in [total_frames // 5, total_frames // 2, total_frames * 3 // 4]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        if len(faces) == 0:
            faces = profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        if len(faces) > len(best_faces):
            best_faces, best_frame = list(faces), frame

    cap.release()

    if best_frame is None:
        return {'error': 'cannot read frame'}

    src_h, src_w = best_frame.shape[:2]
    scale = max(out_w / src_w, out_h / src_h)
    # ceil to even boundary so we always over-scale (never under-fill the output)
    scaled_w = math.ceil(src_w * scale / 2) * 2
    scaled_h = math.ceil(src_h * scale / 2) * 2

    # Default: centre crop
    crop_x = (scaled_w - out_w) // 2
    crop_y = (scaled_h - out_h) // 2
    face_found = False

    if best_faces:
        best_faces = sorted(best_faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = best_faces[0]

        face_cx_s  = (fx + fw / 2) * scale
        face_cy_s  = (fy + fh / 2) * scale
        face_top_s = fy * scale
        face_left_s  = fx * scale
        face_right_s = (fx + fw) * scale

        # Generous headroom above Haar box: Haar starts at forehead, not hairline.
        # 50 % of face height covers hair + safety margin.
        head_top_s = max(0.0, face_top_s - fh * scale * 0.5)

        # Horizontal: centre on face; clamp so face stays inside the crop
        cx = round(face_cx_s - out_w / 2)
        cx = max(0, min(scaled_w - out_w, cx))
        if face_left_s < cx:
            cx = max(0, math.floor(face_left_s))
        if face_right_s > cx + out_w:
            cx = min(scaled_w - out_w, math.ceil(face_right_s - out_w))

        # Vertical: face centre at 35 % from top → head + upper body in frame.
        # Single hard rule: full head (hair included) must be above the crop top.
        cy = round(face_cy_s - out_h * 0.35)
        cy = min(cy, math.floor(head_top_s))   # never start below top-of-head
        cy = max(0, min(scaled_h - out_h, cy))

        crop_x, crop_y = cx, cy
        face_found = True

    return {
        'scaled_w': scaled_w,
        'scaled_h': scaled_h,
        'crop_x': crop_x,
        'crop_y': crop_y,
        'face_found': face_found,
    }


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.serve_file(BASE_DIR / 'SurrealEditor.html', 'text/html')
        elif self.path.startswith('/analyze'):
            qs = parse_qs(urlparse(self.path).query)
            path = qs.get('path', [''])[0]
            w = int(qs.get('w', ['1920'])[0])
            h = int(qs.get('h', ['1080'])[0])
            result = analyze_clip(path, w, h)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        elif self.path.startswith('/progress/'):
            self.stream_progress(self.path[len('/progress/'):])
        elif self.path.startswith('/cancel/'):
            self.cancel_job(self.path[len('/cancel/'):])
        elif self.path.startswith('/files/'):
            fname = Path(self.path[len('/files/'):].replace('%20', ' ')).name
            file_path = Path('/tmp') / fname
            if file_path.exists() and file_path.is_file():
                self.serve_video(file_path)
            else:
                self.send_error(404)
        else:
            rel = self.path.lstrip('/').replace('%20', ' ')
            file_path = BASE_DIR / rel
            if file_path.exists() and file_path.is_file():
                self.serve_video(file_path)
            else:
                self.send_error(404)

    def do_POST(self):
        if self.path == '/run':
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            cmd = data['cmd']
            job_id = str(uuid.uuid4())

            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            jobs[job_id] = {'process': proc, 'lines': [], 'done': False, 'returncode': None}

            def reader():
                for line in proc.stdout:
                    jobs[job_id]['lines'].append(line.rstrip())
                proc.wait()
                jobs[job_id]['done'] = True
                jobs[job_id]['returncode'] = proc.returncode

            threading.Thread(target=reader, daemon=True).start()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'job_id': job_id}).encode())
        elif self.path.startswith('/upload'):
            qs = parse_qs(urlparse(self.path).query)
            filename = Path(qs.get('name', ['upload'])[0]).name
            length = int(self.headers.get('Content-Length', 0))
            data = self.rfile.read(length)
            dest = UPLOAD_DIR / filename
            dest.write_bytes(data)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'path': str(dest)}).encode())
        else:
            self.send_error(404)

    def stream_progress(self, job_id):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        sent = 0
        while True:
            job = jobs.get(job_id)
            if not job:
                break
            lines = job['lines']
            while sent < len(lines):
                msg = lines[sent].replace('\n', ' ')
                self.wfile.write(f'data: {msg}\n\n'.encode())
                sent += 1
            if job['done']:
                rc = job['returncode']
                self.wfile.write(f'event: done\ndata: {rc}\n\n'.encode())
                try:
                    self.wfile.flush()
                except Exception:
                    pass
                break
            try:
                self.wfile.flush()
            except Exception:
                break
            time.sleep(0.1)

    def cancel_job(self, job_id):
        job = jobs.get(job_id)
        if job and not job['done']:
            try:
                job['process'].terminate()
            except Exception:
                pass
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def serve_file(self, path, content_type):
        try:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, str(e))

    def serve_video(self, path):
        size = path.stat().st_size
        range_header = self.headers.get('Range')

        if range_header:
            rng = range_header.strip().replace('bytes=', '')
            start_s, end_s = rng.split('-')
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
            length = end - start + 1

            self.send_response(206)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.send_header('Content-Length', str(length))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            with open(path, 'rb') as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Length', str(size))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            with open(path, 'rb') as f:
                self.wfile.write(f.read())


if __name__ == '__main__':
    server = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    print(f'Surreal Editor  →  http://0.0.0.0:{PORT}')
    print('Press Ctrl+C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
