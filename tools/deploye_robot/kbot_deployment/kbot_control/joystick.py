import json
import socket
from inputs import get_gamepad
from rich.live import Live
from rich.table import Table
import threading
from dataclasses import dataclass
from Controller import Controller, ButtonState
import time

@dataclass
class ControlVector:
    XVel: float = 0.0
    YVel: float = 0.0
    YawRate: float = 0.0

    def to_msg(self) -> bytes:
        json_str = json.dumps({
            "XVel": self.XVel,
            "YVel": self.YVel,
            "YawRate": self.YawRate
        }) + "\n"

        return json_str.encode('utf-8')

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

    def update_commands_from_controller(self, ctrl: Controller) -> None:
        if ctrl.btns.LB == ButtonState.PRESSED:
            XVel_norm = -ctrl.JOYSTICK_LEFT_Y # Robot forward is positive X, stick down is positive
            YVel_norm = -ctrl.JOYSTICK_LEFT_X # Robot left is positive Y, stick right is positive
            YawRate_norm = -ctrl.JOYSTICK_RIGHT_X # Robot counterclockwise is positive, stick right is positive

            max_cmd = self.max_cmd

            self.cmds = ControlVector(
                XVel=XVel_norm * max_cmd,
                YVel=YVel_norm * max_cmd,
                YawRate=YawRate_norm * max_cmd,
                )

        else:
            self.cmds = ControlVector(
                XVel=0,
                YVel=0,
                YawRate=0,
            )

    def command(self) -> None:
        msg = self.cmds.to_msg()
        self.sock.sendto(msg, (self.UDP_IP, self.UDP_PORT))

class CommandDisplay:
    def __init__(self):
        self.controller: Controller = Controller()
        self.commander: Commander = Commander()

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

        table.add_row("XVel", self.make_bar(cmds.XVel, (min_val, max_val), 100, "red", inverted=False))
        table.add_row("YVel", self.make_bar(cmds.YVel, (min_val, max_val), 100, "green", inverted=True))
        table.add_row("Yaw", self.make_bar(cmds.YawRate, (min_val, max_val), 100, "blue", inverted=True))

        return table

    def run(self):
        thread = threading.Thread(target=self.controller_thread, daemon=True)
        thread.start()

        with Live(self.render_table(self.commander.cmds), refresh_per_second=50, screen=False) as live:
            try:
                while thread.is_alive():
                    self.commander.update_commands_from_controller(self.controller)
                    live.update(self.render_table(self.commander.cmds))
                    self.commander.command()
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass

if __name__ == "__main__":
    CommandDisplay().run()
