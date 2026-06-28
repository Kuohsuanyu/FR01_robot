import json
import socket
from dataclasses import dataclass
import time
import threading
from rich.live import Live
from rich.table import Table

# Reuse the local controller abstraction
from Controller import Controller, ButtonState


@dataclass
class ControlVector16:
    # 0..2 base velocities
    XVel: float = 0.0
    YVel: float = 0.0
    YawRate: float = 0.0

    # 3..5 base pose
    BaseHeight: float = 0.0
    BaseRoll: float = 0.0
    BasePitch: float = 0.0

    # 6..10 right arm
    RShoulderPitch: float = 0.0
    RShoulderRoll: float = 0.0
    RElbowPitch: float = 0.0
    RElbowRoll: float = 0.0
    RWristPitch: float = 0.0

    # 11..15 left arm
    LShoulderPitch: float = 0.0
    LShoulderRoll: float = 0.0
    LElbowPitch: float = 0.0
    LElbowRoll: float = 0.0
    LWristPitch: float = 0.0

    def to_msg(self) -> bytes:
        # Keep the same JSON keys expected by firmware (serde rename fields)
        payload = {
            "XVel": self.XVel,
            "YVel": self.YVel,
            "YawRate": self.YawRate,
            "BaseHeight": self.BaseHeight,
            "BaseRoll": self.BaseRoll,
            "BasePitch": self.BasePitch,
            "RShoulderPitch": self.RShoulderPitch,
            "RShoulderRoll": self.RShoulderRoll,
            "RElbowPitch": self.RElbowPitch,
            "RElbowRoll": self.RElbowRoll,
            "RWristPitch": self.RWristPitch,
            "LShoulderPitch": self.LShoulderPitch,
            "LShoulderRoll": self.LShoulderRoll,
            "LElbowPitch": self.LElbowPitch,
            "LElbowRoll": self.LElbowRoll,
            "LWristPitch": self.LWristPitch,
        }
        return (json.dumps(payload) + "\n").encode("utf-8")


class Commander16:
    def __init__(self, udp_ip: str = "localhost", udp_port: int = 10000):
        self.UDP_IP = udp_ip
        self.UDP_PORT = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.max_cmd = 0.5
        self._ultimate_max = 1.5
        self._ultimate_min = 0.1
        self.cmds = ControlVector16()

    def increase_max_cmd(self) -> None:
        self.max_cmd = min(self.max_cmd + 0.1, self._ultimate_max)

    def decrease_max_cmd(self) -> None:
        self.max_cmd = max(self.max_cmd - 0.1, self._ultimate_min)

    def update_from_controller(self, ctrl: Controller) -> None:
        # Gate on LB held, same behavior as the basic script
        if ctrl.btns.LB == ButtonState.PRESSED:
            x_norm = -ctrl.JOYSTICK_LEFT_Y   # forward is +X, stick down is +
            y_norm = -ctrl.JOYSTICK_LEFT_X   # left is +Y, stick right is +
            yaw_norm = -ctrl.JOYSTICK_RIGHT_X  # CCW +, stick right is +

            m = self.max_cmd
            self.cmds.XVel = x_norm * m
            self.cmds.YVel = y_norm * m
            self.cmds.YawRate = yaw_norm * m

            # Leave the remaining 13 fields at zero for now
        else:
            # All zeros when LB not pressed
            self.cmds = ControlVector16()

    def send(self) -> None:
        self.sock.sendto(self.cmds.to_msg(), (self.UDP_IP, self.UDP_PORT))


class CommandDisplay16:
    def __init__(self, udp_ip: str = "localhost", udp_port: int = 10000):
        self.controller: Controller = Controller()
        self.commander: Commander16 = Commander16(udp_ip, udp_port)

    def controller_thread(self):
        while True:
            if self.controller.rising_edge('UP'):
                self.commander.increase_max_cmd()
            elif self.controller.rising_edge('DOWN'):
                self.commander.decrease_max_cmd()
            self.controller.update()

    def make_bar(self, value: float, valid_range: tuple[float, float], width: int, color: str, inverted=False) -> str:
        min_val, max_val = valid_range
        normalized = (value - min_val) / (max_val - min_val)
        if inverted:
            normalized = 1.0 - normalized
        normalized = max(0.0, min(1.0, normalized))
        filled = int(normalized * width)
        bar = "█" * filled + " " * (width - filled)
        return f"[{color}][{bar}][/{color}] {value:+.2f}"

    def render_table(self, c: ControlVector16) -> Table:
        speed_text = f"[bold bright_yellow on magenta]MAX COMMAND: {round(self.commander.max_cmd, 1)}[/]"
        table = Table(title="Control Vector (16D)", box=None)
        table.add_column("Axis", justify="right", no_wrap=True)
        table.add_column("Value", justify="center")

        min_val = -self.commander._ultimate_max
        max_val = self.commander._ultimate_max

        table.add_row("", speed_text)
        table.add_row("")
        table.add_row("XVel", self.make_bar(c.XVel, (min_val, max_val), 100, "red", inverted=False))
        table.add_row("YVel", self.make_bar(c.YVel, (min_val, max_val), 100, "green", inverted=True))
        table.add_row("Yaw", self.make_bar(c.YawRate, (min_val, max_val), 100, "blue", inverted=True))
        # Show a summary row for the other fields staying at zero
        table.add_row("")
        table.add_row("Others", "13 extended fields = 0.0")
        return table

    def run(self):
        thread = threading.Thread(target=self.controller_thread, daemon=True)
        thread.start()
        with Live(self.render_table(self.commander.cmds), refresh_per_second=50, screen=False) as live:
            try:
                while thread.is_alive():
                    self.commander.update_from_controller(self.controller)
                    live.update(self.render_table(self.commander.cmds))
                    self.commander.send()
                    time.sleep(0.05)  # 20 Hz
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    CommandDisplay16().run()