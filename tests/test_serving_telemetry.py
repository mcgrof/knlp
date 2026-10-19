"""Read actual HTTP metric bytes and preserve collection errors separately."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

from tools.kv.native.serving_telemetry import MetricSampler


def test_raw_queue_metrics_and_clean_thread_shutdown():
    text = b'vllm:num_requests_waiting{model_name="study"} 7\n'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/metrics"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(text)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with MetricSampler(
            f"http://127.0.0.1:{server.server_port}", interval=0.01
        ) as sampler:
            deadline = time.monotonic() + 2
            while not sampler.rows and time.monotonic() < deadline:
                time.sleep(0.01)
        assert not sampler.thread.is_alive()
        assert sampler.rows[0]["prometheus_text"] == text.decode()
        assert sampler.rows[0]["end_ns"] >= sampler.rows[0]["start_ns"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
