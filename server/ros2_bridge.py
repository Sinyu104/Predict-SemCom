"""
ros2_bridge.py  —  Thin ROS2 ↔ TCP bridge (Python 3.10, system python).

Forwards /vla/request messages to the PI0Fast inference server (vla_server.py)
over a persistent local TCP socket (connect once, reuse for all requests).

Run this with the system python3 (which has rclpy):
    export ROS_DOMAIN_ID=66
    export ROS_DISCOVERY_SERVER=10.32.33.49:11811
    source ~/ros2_humble/ros2-linux/setup.bash
    python3 server/ros2_bridge.py

Run vla_server.py first with the worldmodel conda python (Python 3.12):
    /path/to/worldmodel/bin/python -u server/vla_server.py --lora_path ...
"""

import argparse
import json
import socket
import sys
import threading

try:
    import rclpy
    from std_msgs.msg import String
except ImportError:
    print("[Bridge] ERROR: rclpy not found. Source your ROS2 workspace.")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="ROS2 ↔ TCP bridge for PI0Fast server")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    return p.parse_args()


def recv_all(sock, n):
    """Read exactly n bytes from sock."""
    chunks, received = [], 0
    while received < n:
        chunk = sock.recv(min(65536, n - received))
        if not chunk:
            raise ConnectionError("Inference server disconnected")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def main():
    args = parse_args()

    # Connect once — persistent for all requests
    print(f"[Bridge] Connecting to inference server at {args.host}:{args.port} …")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[Bridge] Connected.")

    # Mutex so concurrent ROS2 callbacks don't interleave on the socket
    lock = threading.Lock()

    rclpy.init()
    node = rclpy.create_node("vla_bridge")
    pub  = node.create_publisher(String, "/vla/response", 10)

    def on_request(msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            node.get_logger().error(f"Malformed request JSON: {e}")
            return

        seq     = data.get("seq", 0)
        payload = json.dumps(data).encode()

        try:
            with lock:
                sock.sendall(len(payload).to_bytes(4, "big") + payload)
                resp_len = int.from_bytes(recv_all(sock, 4), "big")
                response = json.loads(recv_all(sock, resp_len))
        except Exception as e:
            node.get_logger().error(f"Inference server error: {e}")
            response = {"seq": seq, "action": [[0.0] * 7],
                        "status": "error", "message": str(e)}

        reply      = String()
        reply.data = json.dumps(response)
        pub.publish(reply)
        node.get_logger().info(
            f"seq={seq}  status={response.get('status')}  "
            f"chunk_len={len(response.get('action', []))}"
        )

    node.create_subscription(String, "/vla/request", on_request, 10)

    print(f"[Bridge] ROS2 node started.")
    print(f"[Bridge] Subscribed to  /vla/request")
    print(f"[Bridge] Publishing to  /vla/response")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[Bridge] Shutting down.")
    finally:
        sock.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
