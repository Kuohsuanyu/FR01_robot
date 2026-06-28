import json
import socket
import time
import threading
from dataclasses import dataclass
from typing import Tuple
import os
import sys
import select
import termios
import tty
import atexit

from inputs import get_key
from rich.live import Live
from rich.table import Table


@dataclass
class ControlVector:
    XVel: float = 0.0
    YVel: float = 0.0
    YawRate: float = 0.0

    def to_msg(self) -> bytes:
        json_str = (
            json.dumps({"XVel": self.XVel, "YVel": self.YVel, "YawRate": self.YawRate})
            + "\n"
        )
        return json_str.encode("utf-8")


class Commander:
    def __init__(self):
        self.UDP_IP = "localhost"
        self.UDP_PORT = 10000
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.max_cmd = 0.5
        self._ultimate_max = 1.5
        self._ultimate_min = 0.1
        self.cmds = ControlVector()

    def increase_max_cmd(self) -> None:
        self.max_cmd = min(self.max_cmd + 0.1, self._ultimate_max)

    def decrease_max_cmd(self) -> None:
        self.max_cmd = max(self.max_cmd - 0.1, self._ultimate_min)

    def update_commands_from_keyboard(self, kb_state: "KeyboardState") -> None:
        x_norm, y_norm, yaw_norm = kb_state.get_normalized_axes()
        max_cmd = self.max_cmd

        self.cmds = ControlVector(
            XVel=x_norm * max_cmd,
            YVel=y_norm * max_cmd,
            YawRate=yaw_norm * max_cmd,
        )

    def command(self) -> None:
        msg = self.cmds.to_msg()
        self.sock.sendto(msg, (self.UDP_IP, self.UDP_PORT))


class KeyboardState:
    """Tracks current pressed keys and computes normalized control axes.

    Mappings:
    - W/S -> XVel (+1 forward with W, -1 with S)
    - A/D -> YVel (+1 left with A, -1 right with D)
    - Q/E -> YawRate (+1 CCW with Q, -1 CW with E)
    """

    KEY_W = "KEY_W"
    KEY_S = "KEY_S"
    KEY_A = "KEY_A"
    KEY_D = "KEY_D"
    KEY_Q = "KEY_Q"
    KEY_E = "KEY_E"
    KEY_UP = "KEY_UP"
    KEY_DOWN = "KEY_DOWN"

    def __init__(self) -> None:
        self._pressed_codes: set[str] = set()
        self._lock = threading.Lock()

    def press(self, code: str) -> None:
        with self._lock:
            self._pressed_codes.add(code)

    def release(self, code: str) -> None:
        with self._lock:
            if code in self._pressed_codes:
                self._pressed_codes.remove(code)

    def _is_pressed_unlocked(self, code: str) -> bool:
        return code in self._pressed_codes

    def is_pressed(self, code: str) -> bool:
        with self._lock:
            return self._is_pressed_unlocked(code)

    def get_normalized_axes(self) -> Tuple[float, float, float]:
        with self._lock:
            x = 0.0
            y = 0.0
            yaw = 0.0

            if self._is_pressed_unlocked(self.KEY_W):
                x += 1.0
            if self._is_pressed_unlocked(self.KEY_S):
                x -= 1.0

            # Robot left is positive Y
            if self._is_pressed_unlocked(self.KEY_A):
                y += 1.0
            if self._is_pressed_unlocked(self.KEY_D):
                y -= 1.0

            # Robot CCW is positive yaw
            if self._is_pressed_unlocked(self.KEY_Q):
                yaw += 1.0
            if self._is_pressed_unlocked(self.KEY_E):
                yaw -= 1.0

            # Clamp to [-1, 1]
            x = max(-1.0, min(1.0, x))
            y = max(-1.0, min(1.0, y))
            yaw = max(-1.0, min(1.0, yaw))

            return x, y, yaw


class CommandDisplay:
    def __init__(self):
        self.keyboard_state = KeyboardState()
        self.commander = Commander()
        self._last_adjust_time = 0.0
        self._adjust_interval_s = 0.18
        self.last_event: str = ""
        self.last_error: str = ""
        self._debug = os.environ.get("KBOT_KEY_DEBUG", "").lower() in ("1", "true", "yes")

        # Input mode selection: 'evdev', 'stdin', 'auto'
        env_mode = os.environ.get("KBOT_INPUT_MODE", "auto").lower()
        if env_mode not in ("evdev", "stdin", "auto"):
            env_mode = "auto"
        if env_mode == "auto":
            if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
                self.input_mode = "stdin"
            else:
                self.input_mode = "evdev"
        else:
            self.input_mode = env_mode

        # State used for stdin mode
        self._stdin_fd: int | None = None
        self._stdin_old_attrs: tuple | None = None
        self._stdin_buf: str = ""
        self._key_last_seen: dict[str, float] = {}
        self._repeat_release_s: float = 0.25

    def _maybe_adjust_speed(self, pressed_up: bool, pressed_down: bool) -> None:
        now = time.time()
        if now - self._last_adjust_time < self._adjust_interval_s:
            return
        if pressed_up:
            self.commander.increase_max_cmd()
            self._last_adjust_time = now
        elif pressed_down:
            self.commander.decrease_max_cmd()
            self._last_adjust_time = now

    def keyboard_thread(self) -> None:
        # Blocking loop reading key events from /dev/input
        while True:
            try:
                events = get_key()
            except PermissionError as e:
                self.last_error = f"PermissionError: {e}. Add user to 'input' group or run with sudo."
                time.sleep(0.5)
                continue
            except Exception as e:
                # Keep running even if transient read errors happen
                self.last_error = f"{type(e).__name__}: {e}"
                time.sleep(0.1)
                continue

            for event in events:
                code = event.code
                state = event.state  # 1=pressed, 0=released, 2=hold/auto-repeat

                if state == 1:  # pressed
                    self.keyboard_state.press(code)
                    self.last_event = f"{code} DOWN"
                    if self._debug:
                        print(self.last_event)
                    if code == KeyboardState.KEY_UP:
                        self._maybe_adjust_speed(pressed_up=True, pressed_down=False)
                    elif code == KeyboardState.KEY_DOWN:
                        self._maybe_adjust_speed(pressed_up=False, pressed_down=True)
                elif state == 0:  # released
                    self.keyboard_state.release(code)
                    self.last_event = f"{code} UP"
                    if self._debug:
                        print(self.last_event)
                # ignore state == 2 (hold)

    # ======== STDIN (SSH) backend ========
    def _enter_raw_mode(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("stdin is not a TTY; cannot use stdin mode")
        self._stdin_fd = sys.stdin.fileno()
        self._stdin_old_attrs = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        # make stdin non-blocking via select; no need to change flags
        atexit.register(self._exit_raw_mode)

    def _exit_raw_mode(self) -> None:
        if self._stdin_fd is not None and self._stdin_old_attrs is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_attrs)
            except Exception:
                pass

    def _stdin_mark_pressed(self, code: str) -> None:
        self.keyboard_state.press(code)
        self._key_last_seen[code] = time.time()
        self.last_event = f"{code} DOWN"
        if self._debug:
            print(self.last_event)

    def _stdin_expire_keys(self) -> None:
        now = time.time()
        to_release: list[str] = []
        for code, last_seen in list(self._key_last_seen.items()):
            if now - last_seen > self._repeat_release_s:
                to_release.append(code)
        for code in to_release:
            self.keyboard_state.release(code)
            self._key_last_seen.pop(code, None)

    def _process_stdin_buffer(self) -> None:
        # Process known escape sequences and printable keys
        s = self._stdin_buf
        # Handle arrow keys
        while True:
            idx = s.find("\x1b[")
            if idx == -1:
                break
            if len(s) < idx + 3:
                break  # incomplete sequence; wait for more
            seq = s[idx : idx + 3]
            if seq == "\x1b[A":
                self._stdin_mark_pressed(KeyboardState.KEY_UP)
                s = s[:idx] + s[idx + 3 :]
            elif seq == "\x1b[B":
                self._stdin_mark_pressed(KeyboardState.KEY_DOWN)
                s = s[:idx] + s[idx + 3 :]
            else:
                # Unused arrows or other sequences; drop three chars
                s = s[:idx] + s[idx + 3 :]

        # Handle letters
        for ch in list(s):
            if ch.lower() == "w":
                self._stdin_mark_pressed(KeyboardState.KEY_W)
            elif ch.lower() == "s":
                self._stdin_mark_pressed(KeyboardState.KEY_S)
            elif ch.lower() == "a":
                self._stdin_mark_pressed(KeyboardState.KEY_A)
            elif ch.lower() == "d":
                self._stdin_mark_pressed(KeyboardState.KEY_D)
            elif ch.lower() == "q":
                self._stdin_mark_pressed(KeyboardState.KEY_Q)
            elif ch.lower() == "e":
                self._stdin_mark_pressed(KeyboardState.KEY_E)

        # Anything else is ignored; clear buffer
        self._stdin_buf = ""

    def stdin_thread(self) -> None:
        try:
            self._enter_raw_mode()
        except Exception as e:
            self.last_error = f"stdin mode unavailable: {e}"
            return

        try:
            while True:
                # Read available bytes without blocking using select
                try:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                except Exception:
                    rlist = []
                if rlist:
                    try:
                        data = os.read(sys.stdin.fileno(), 64)
                        if data:
                            try:
                                self._stdin_buf += data.decode("utf-8", errors="ignore")
                            except Exception:
                                # Fallback to latin-1 if necessary
                                self._stdin_buf += data.decode("latin-1", errors="ignore")
                            self._process_stdin_buffer()
                    except Exception as e:
                        self.last_error = f"stdin read error: {e}"

                # Expire keys when repeat stops
                self._stdin_expire_keys()

                # Handle speed adjustments if up/down was recently seen
                if self.keyboard_state.is_pressed(KeyboardState.KEY_UP):
                    self._maybe_adjust_speed(pressed_up=True, pressed_down=False)
                elif self.keyboard_state.is_pressed(KeyboardState.KEY_DOWN):
                    self._maybe_adjust_speed(pressed_up=False, pressed_down=True)
        finally:
            self._exit_raw_mode()

    def make_bar(
        self, value: float, valid_range: Tuple[float, float], width: int, color: str, inverted: bool = False
    ) -> str:
        min_val, max_val = valid_range
        normalized = (value - min_val) / (max_val - min_val)
        if inverted:
            normalized = 1.0 - normalized
        normalized = max(0.0, min(1.0, normalized))
        filled = int(normalized * width)
        bar = "█" * filled + " " * (width - filled)
        return f"[{color}][{bar}][/{color}] {value:+.2f}"

    def render_table(self, cmds: ControlVector) -> Table:
        speed_text = f"[bold bright_yellow on magenta]MAX COMMAND: {round(self.commander.max_cmd, 1)}[/]"

        table = Table(title=f"Control Vector", box=None)

        table.add_column("Axis", justify="right", no_wrap=True)
        table.add_column("Value", justify="center")

        min_val = -self.commander._ultimate_max
        max_val = self.commander._ultimate_max

        table.add_row("", speed_text)
        table.add_row("")

        table.add_row("XVel", self.make_bar(self.commander.cmds.XVel, (min_val, max_val), 100, "red", inverted=False))
        table.add_row("YVel", self.make_bar(self.commander.cmds.YVel, (min_val, max_val), 100, "green", inverted=True))
        table.add_row("Yaw", self.make_bar(self.commander.cmds.YawRate, (min_val, max_val), 100, "blue", inverted=True))
        table.add_row("Mode", f"[magenta]{self.input_mode}[/]")
        table.add_row("LastKey", f"[cyan]{self.last_event or '-'}[/]")
        table.add_row("Error", f"[red]{self.last_error or '-'}[/]")

        return table

    def run(self) -> None:
        if self.input_mode == "stdin":
            thread = threading.Thread(target=self.stdin_thread, daemon=True)
        else:
            thread = threading.Thread(target=self.keyboard_thread, daemon=True)
        thread.start()

        with Live(self.render_table(self.commander.cmds), refresh_per_second=50, screen=False) as live:
            try:
                while thread.is_alive():
                    self.commander.update_commands_from_keyboard(self.keyboard_state)
                    live.update(self.render_table(self.commander.cmds))
                    self.commander.command()
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    CommandDisplay().run()

