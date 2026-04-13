"""
test_ros2_client.py  —  ROS2 connection test for the Windows desktop side.

Does NOT require Isaac Sim. Sends N dummy requests to /vla/request and
checks that matching responses arrive on /vla/response.

Run this with the system Python (or any Python that has rclpy), NOT
through Isaac Sim's python.bat — it has no Isaac Sim dependency.

Usage (Windows cmd):
    set ROS_DOMAIN_ID=66
    set ROS_DISCOVERY_SERVER=10.32.33.41:11811
    python isaac_sim\test_ros2_client.py --num_requests 5

Expected output when the Linux test server is running:
    [TestClient] ROS2 node started.
    [TestClient] Sending request seq=1 ...
    [TestClient] Response seq=1  action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  latency=23.4 ms  OK
    [TestClient] Sending request seq=2 ...
    ...
    [TestClient] All 5 requests succeeded.  avg_latency=24.1 ms
"""

import argparse
import base64
import io
import json
import os
import sys
import threading
import time


def make_dummy_jpeg(width: int = 224, height: int = 224) -> str:
    """Create a small solid-colour JPEG and return as base64 string."""
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(128, 64, 32))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        # Pillow not available — send a 1-byte placeholder so the test still runs
        print("[TestClient] WARNING: Pillow not installed — sending empty image placeholder.")
        return base64.b64encode(b"\xff\xd8\xff\xd9").decode("utf-8")   # minimal JPEG


def main():
    parser = argparse.ArgumentParser(description="ROS2 VLA connection test (client side)")
    parser.add_argument("--num_requests", type=int, default=5,
                        help="Number of test requests to send (default: 5)")
    parser.add_argument("--timeout_s",   type=float, default=10.0,
                        help="Seconds to wait for each response (default: 10)")
    args = parser.parse_args()

    try:
        import rclpy
        from std_msgs.msg import String
    except ImportError:
        print("[TestClient] ERROR: rclpy not found.")
        print("  On Windows: install ROS2 and source the setup script, or")
        print("  use WSL2 with ROS2 installed.")
        sys.exit(1)

    print("[TestClient] ROS2 connection test")
    print(f"[TestClient] ROS_DOMAIN_ID        = {os.environ.get('ROS_DOMAIN_ID', '(not set ← set to 66)')}")
    print(f"[TestClient] ROS_DISCOVERY_SERVER = {os.environ.get('ROS_DISCOVERY_SERVER', '(not set ← set to 10.32.33.41:11811)')}")
    print()

    # ── ROS2 setup ──────────────────────────────────────────────────────── #
    rclpy.init()
    node = rclpy.create_node("vla_test_client")

    pending  = {}   # seq → threading.Event
    results  = {}   # seq → reply dict
    lock     = threading.Lock()

    def on_response(msg: String):
        try:
            data = json.loads(msg.data)
            seq  = data.get("seq", -1)
            with lock:
                if seq in pending:
                    results[seq] = data
                    pending[seq].set()
        except json.JSONDecodeError as e:
            print(f"[TestClient] Malformed response JSON: {e}")

    pub = node.create_publisher(String, "/vla/request", 10)
    node.create_subscription(String, "/vla/response", on_response, 10)

    executor    = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("[TestClient] ROS2 node started.")
    print("[TestClient] Waiting 1 s for discovery server to register node …")
    time.sleep(1.0)   # give Fast DDS time to discover the server node

    # ── Send test requests ───────────────────────────────────────────────── #
    jpeg_b64    = make_dummy_jpeg()
    latencies   = []
    n_success   = 0

    for i in range(1, args.num_requests + 1):
        seq   = i
        event = threading.Event()
        with lock:
            pending[seq] = event

        msg      = String()
        msg.data = json.dumps({
            "seq":         seq,
            "jpeg_b64":    jpeg_b64,
            "instruction": "test connection",
            "unnorm_key":  "franka_isaac",
        })

        t0 = time.perf_counter()
        pub.publish(msg)
        print(f"[TestClient] Sending request seq={seq} …", end=" ", flush=True)

        if event.wait(timeout=args.timeout_s):
            ms = (time.perf_counter() - t0) * 1000
            with lock:
                reply = results.pop(seq)
                pending.pop(seq, None)

            action = reply.get("action", [])
            status = reply.get("status", "?")
            print(f"action={action}  latency={ms:.1f} ms  {status.upper()}")
            latencies.append(ms)
            n_success += 1
        else:
            with lock:
                pending.pop(seq, None)
            print(f"TIMEOUT after {args.timeout_s}s — server not reachable")
            print("  Check:")
            print("    1. test_ros2_server.py is running on the Linux server")
            print("    2. ROS_DOMAIN_ID matches on both machines (should be 66)")
            print("    3. ROS_DISCOVERY_SERVER matches on both machines (10.32.33.41:11811)")
            print("    4. Firewall allows TCP port 11811")

        time.sleep(0.2)   # small gap between requests

    # ── Summary ─────────────────────────────────────────────────────────── #
    print()
    if n_success == args.num_requests:
        avg = sum(latencies) / len(latencies)
        print(f"[TestClient] All {n_success}/{args.num_requests} requests succeeded.  avg_latency={avg:.1f} ms")
        print("[TestClient] Connection OK — you can now run isaac_collector.py")
    else:
        print(f"[TestClient] {n_success}/{args.num_requests} requests succeeded.  Connection has issues.")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
