#!/usr/bin/env python3
import http.server
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

PORT = int(os.environ.get('PORT', 8877))
BASE_DIR = Path(__file__).parent

jobs = {}  # job_id -> {process, lines, done, returncode}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.serve_file(BASE_DIR / 'SurrealEditor.html', 'text/html')
        elif self.path.startswith('/progress/'):
            self.stream_progress(self.path[len('/progress/'):])
        elif self.path.startswith('/cancel/'):
            self.cancel_job(self.path[len('/cancel/'):])
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
    server = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
    print(f'Surreal Editor  →  http://0.0.0.0:{PORT}')
    print('Press Ctrl+C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
