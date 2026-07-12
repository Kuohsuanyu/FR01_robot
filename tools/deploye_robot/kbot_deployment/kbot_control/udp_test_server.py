import argparse
import json
import socket
import sys
import time
from datetime import datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UDP test server: prints received datagrams")
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=10000, help="UDP port to bind (default: 10000)")
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not attempt to parse payload as JSON; print raw only",
    )
    return parser.parse_args()


def format_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def main() -> int:
    args = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Allow quick rebinding on restart
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))

    print(f"[{format_ts()}] Listening on {args.host}:{args.port} (Ctrl+C to exit)")

    try:
        while True:
            try:
                data, addr = sock.recvfrom(8192)
            except KeyboardInterrupt:
                print("\nInterrupted. Exiting...")
                return 0
            except Exception as e:
                print(f"[{format_ts()}] recv error: {e}")
                time.sleep(0.05)
                continue

            try:
                text = data.decode("utf-8", errors="replace").rstrip("\n")
            except Exception:
                text = str(data)

            print(f"[{format_ts()}] From {addr[0]}:{addr[1]} -> {text}")

            if not args.no_json:
                try:
                    obj = json.loads(text)
                    print(f"  parsed: {json.dumps(obj, indent=2)}")
                except json.JSONDecodeError:
                    # Not JSON; it's fine
                    pass
    finally:
        try:
            sock.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

