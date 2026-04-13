"""
test_ros2_server.py  —  ROS2 connection test for the Linux server side.

Does NOT load OpenVLA. Subscribes to /vla/request, prints what arrives,
and immediately replies with a dummy zero action.

Usage (Linux server):
    export ROS_DOMAIN_ID=66
    export ROS_DISCOVERY_SERVER=10.32.33.41:11811
    source /opt/ros/humble/setup.bash
    python server/test_ros2_server.py

Expected output when the Windows test client sends a request:
    [TestServer] Received request  seq=1  instruction='test connection'
    [TestServer] Image: 224x224 RGB  (JPEG bytes decoded OK)
    [TestServer] Replied with zero action  seq=1
"""

import base64
import io
import json
import sys
import time

try:
    import rclpy
    from std_msgs.msg import String
except ImportError:
    print("[TestServer] ERROR: rclpy not found.")
    print("  Run:  source /opt/ros/humble/setup.bash")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None


def main():
    rclpy.init()
    node = rclpy.create_node("vla_test_server")
    pub  = node.create_publisher(String, "/vla/response", 10)

    n_received = 0

    def on_request(msg: String):
        nonlocal n_received
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            node.get_logger().error(f"Malformed JSON: {e}")
            return

        seq         = data.get("seq", -1)
        instruction = data.get("instruction", "(none)")
        jpeg_b64    = data.get("jpeg_b64", "")

        print(f"\n[TestServer] Received request  seq={seq}  instruction='{instruction}'")

        # Decode and inspect the image
        if jpeg_b64:
            try:
                jpeg_bytes = base64.b64decode(jpeg_b64)
                if Image is not None:
                    img = Image.open(io.BytesIO(jpeg_bytes))
                    print(f"[TestServer] Image: {img.width}x{img.height} {img.mode}  ({len(jpeg_bytes)} JPEG bytes — OK)")
                else:
                    print(f"[TestServer] Image: {len(jpeg_bytes)} JPEG bytes received  (install Pillow to decode)")
            except Exception as e:
                print(f"[TestServer] Image decode ERROR: {e}")
        else:
            print("[TestServer] WARNING: no jpeg_b64 field in request")

        # Reply with a dummy zero action
        reply      = String()
        reply.data = json.dumps({
            "seq":    seq,
            "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "status": "ok",
        })
        pub.publish(reply)
        n_received += 1
        print(f"[TestServer] Replied with zero action  seq={seq}  (total requests: {n_received})")

    node.create_subscription(String, "/vla/request", on_request, 10)

    print("[TestServer] ROS2 test server started.")
    print("[TestServer] Subscribed to /vla/request")
    print("[TestServer] Publishing to  /vla/response")
    print("[TestServer] Waiting for requests from the Windows desktop …")
    print(f"[TestServer] ROS_DOMAIN_ID      = {__import__('os').environ.get('ROS_DOMAIN_ID', '(not set)')}")
    print(f"[TestServer] ROS_DISCOVERY_SERVER = {__import__('os').environ.get('ROS_DISCOVERY_SERVER', '(not set)')}")
    print()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"\n[TestServer] Stopped after {n_received} requests.")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
