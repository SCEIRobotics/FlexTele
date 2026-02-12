import argparse
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import zenoh


FRONTEND_DIR = Path(__file__).parent / "frontend"


class SharedFrame:
    def __init__(self):
        self._frame = None
        self._seq = 0
        self._cv = threading.Condition()

    def update(self, jpeg_bytes: bytes):
        with self._cv:
            self._frame = jpeg_bytes
            self._seq += 1
            self._cv.notify_all()

    def wait_next(self, last_seq: int, timeout: float = 5.0):
        with self._cv:
            if self._seq == last_seq:
                self._cv.wait(timeout=timeout)
            return self._seq, self._frame


class ImageBridge:
    def __init__(self, key_expr: str, frame_store: SharedFrame, swap_rb: bool, jpeg_quality: int):
        self.frame_store = frame_store
        self.swap_rb = swap_rb
        self.jpeg_quality = max(1, min(100, jpeg_quality))
        self.session = zenoh.open(zenoh.Config())
        self.session.declare_subscriber(key_expr, self._on_sample)

    def _on_sample(self, sample: zenoh.Sample):
        payload = sample.payload.to_bytes()
        if not payload:
            return

        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return

        # Some upstream pipelines publish JPEG with RGB channel order assumptions.
        # Browser MJPEG expects correctly encoded color, so we normalize here.
        if self.swap_rb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                self.jpeg_quality,
                int(cv2.IMWRITE_JPEG_OPTIMIZE),
                1,
            ],
        )
        if ok:
            self.frame_store.update(encoded.tobytes())


def make_handler(frame_store: SharedFrame):
    class StreamHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                return self._serve_static("index.html")

            if path.startswith("/static/"):
                rel_path = path.replace("/static/", "", 1)
                return self._serve_static(rel_path)

            if path == "/stream.mjpg":
                return self._serve_stream()

            self.send_error(404, "Not Found")

        def _serve_static(self, rel_path: str):
            safe_path = (FRONTEND_DIR / rel_path).resolve()
            if not str(safe_path).startswith(str(FRONTEND_DIR.resolve())):
                self.send_error(403, "Forbidden")
                return
            if not safe_path.exists() or not safe_path.is_file():
                self.send_error(404, "Not Found")
                return

            data = safe_path.read_bytes()
            content_type, _ = mimetypes.guess_type(str(safe_path))
            self.send_response(200)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _serve_stream(self):
            self.send_response(200)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            seq = 0
            while True:
                seq, frame = frame_store.wait_next(seq, timeout=5.0)
                if frame is None:
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break

        def log_message(self, fmt, *args):
            return

    return StreamHandler


def main():
    parser = argparse.ArgumentParser(description="Web viewer for Isaac camera stream over Zenoh")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8080, help="HTTP bind port")
    parser.add_argument("--topic", default="isaac/head_cam", help="Zenoh key expression")
    parser.add_argument(
        "--swap-rb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Swap R/B channels before re-encoding (default: enabled)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=98,
        help="Output JPEG quality for MJPEG stream, 1-100 (default: 98)",
    )
    args = parser.parse_args()

    if not FRONTEND_DIR.exists():
        raise FileNotFoundError(f"frontend directory not found: {FRONTEND_DIR}")

    frame_store = SharedFrame()
    ImageBridge(
        key_expr=args.topic,
        frame_store=frame_store,
        swap_rb=args.swap_rb,
        jpeg_quality=args.jpeg_quality,
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(frame_store))
    print(
        f"[web_isaac_image] open http://{args.host}:{args.port} "
        f"topic={args.topic} swap_rb={args.swap_rb} jpeg_quality={args.jpeg_quality}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
