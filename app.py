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
    except ImportError:
        return {'error': 'opencv not available'}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'cannot open video'}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 4))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {'error': 'cannot read frame'}

    src_h, src_w = frame.shape[:2]
    scale = max(out_w / src_w, out_h / src_h)
    # round to even pixel boundary (H.264/H.265 requirement)
    scaled_w = round(src_w * scale / 2) * 2
    scaled_h = round(src_h * scale / 2) * 2

    crop_x = (scaled_w - out_w) // 2
    crop_y = (scaled_h - out_h) // 2
    face_found = False

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]

            face_top_s    = fy * scale
            face_bottom_s = (fy + fh) * scale
            face_left_s   = fx * scale
            face_right_s  = (fx + fw) * scale
            face_cx_s     = (fx + fw / 2) * scale
            face_cy_s     = (fy + fh / 2) * scale
            headroom      = fh * scale * 0.35
            chin_room     = fh * scale * 0.1

            # Horizontal: center on face, clamp to keep face in frame
            cx = max(0, min(scaled_w - out_w, round(face_cx_s - out_w / 2)))
            if face_left_s - cx < 0:
                cx = max(0, round(face_left_s))
            if face_right_s - cx > out_w:
                cx = min(scaled_w - out_w, round(face_right_s - out_w))

            # Vertical: face center at 33% from top (head + upper body framing)
            cy_ideal = round(face_cy_s - out_h * 0.33)
            cy_min = round(face_bottom_s + chin_room - out_h)   # must show chin
            cy_max = round(face_top_s - headroom)               # must show head
            if cy_min > cy_max:
                cy = cy_max  # face taller than frame — prioritise head
            else:
                cy = max(cy_min, min(cy_max, cy_ideal))
            cy = max(0, min(scaled_h - out_h, cy))

            crop_x = cx
            crop_y = cy
            face_found = True
    except Exception:
        pass

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
