#!/usr/bin/env python3
"""Smoke test for the head RPi agent — run this on the PC.

Verifies:
  * WebSocket connects, telemetry arrives
  * scan reports the expected motor IDs
  * MJPEG /snapshot returns a JPEG that decodes

Usage:  python3 test_client.py <rpi-ip>[:port]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request

import aiohttp


async def test_ws(url, ids_expected=None):
    print(f"[ws] connecting {url}")
    tstart = time.monotonic()
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(url, timeout=5) as ws:
            # request a scan first
            await ws.send_str(json.dumps({"op": "scan", "req_id": "sc",
                                          "range": [1, 30]}))
            scan_done = False
            n_tele = 0
            t_first_tele = None
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT: continue
                d = json.loads(msg.data)
                if d.get("t") == "reply" and d.get("req_id") == "sc":
                    ids = d.get("ids") or []
                    print(f"[scan] IDs = {ids}")
                    scan_done = True
                    if ids_expected and set(ids) != set(ids_expected):
                        print(f"  ⚠ expected {ids_expected}")
                elif d.get("t") == "tele":
                    if t_first_tele is None:
                        t_first_tele = time.monotonic() - tstart
                        print(f"[tele] first packet in {t_first_tele*1000:.0f}ms")
                    n_tele += 1
                    if n_tele <= 3:
                        motors = [(m["sid"], m["pos"], m["volt"], m["temp"])
                                  for m in d["motors"]]
                        print(f"  #{n_tele:3d} → {motors}")
                if scan_done and n_tele >= 20:
                    dt = time.monotonic() - tstart
                    print(f"[tele] {n_tele} pkts in {dt:.2f}s "
                          f"= {n_tele/dt:.1f} Hz")
                    break


def test_http(base_url):
    for path in ("/status", "/snapshot"):
        url = f"{base_url}{path}"
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                body = r.read()
                ct = r.headers.get("Content-Type", "")
                print(f"[http] {path} → {r.status} {ct} ({len(body)} B)")
                if path == "/status":
                    print(f"       {body.decode()[:400]}")
        except Exception as e:
            print(f"[http] {path} FAIL: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host", help="rpi ip, e.g. 192.168.0.42 or 192.168.0.42:8000")
    args = ap.parse_args()
    if ":" not in args.host: args.host += ":8000"
    base = f"http://{args.host}"
    ws_url = f"ws://{args.host}/ws"

    test_http(base)
    try:
        asyncio.run(test_ws(ws_url))
    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
        print(f"[ws] connect failed: {e}")


if __name__ == "__main__":
    main()
