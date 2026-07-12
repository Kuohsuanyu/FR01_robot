from enum import Enum, auto
from types import SimpleNamespace
from inputs import get_gamepad
import asyncio


class ButtonState(Enum):
    PRESSED = auto()
    RELEASED = auto()

class Controller:
    def __init__(self):
        self.btns = SimpleNamespace(
            A=ButtonState.RELEASED,
            B=ButtonState.RELEASED,
            X=ButtonState.RELEASED,
            Y=ButtonState.RELEASED,
            LEFT=ButtonState.RELEASED,
            RIGHT=ButtonState.RELEASED,
            UP=ButtonState.RELEASED,
            DOWN=ButtonState.RELEASED,
            START=ButtonState.RELEASED,
            BACK=ButtonState.RELEASED,
            LB=ButtonState.RELEASED,
            RB=ButtonState.RELEASED,
            MODE=ButtonState.RELEASED,
        )
        self.prev_btns = SimpleNamespace(**self.btns.__dict__)
        self.TRIGGER_LEFT: float = 0.0
        self.TRIGGER_RIGHT: float = 0.0
        self.JOYSTICK_LEFT_X: float = 0.0
        self.JOYSTICK_LEFT_Y: float = 0.0
        self.JOYSTICK_RIGHT_X: float = 0.0
        self.JOYSTICK_RIGHT_Y: float = 0.0

    def rising_edge(self, button: str) -> bool:
        return (getattr(self.btns, button) == ButtonState.PRESSED and 
                getattr(self.prev_btns, button) == ButtonState.RELEASED)

    def falling_edge(self, button: str) -> bool:
        return (getattr(self.btns, button) == ButtonState.RELEASED and 
                getattr(self.prev_btns, button) == ButtonState.PRESSED)

    def update(self):
        # Save previous states before processing new events
        self.prev_btns = SimpleNamespace(**self.btns.__dict__)
        events = get_gamepad()
        for event in events: 
            if event.code == "ABS_Y":
                self.JOYSTICK_LEFT_Y = self._clean_joystick(event.state)
            elif event.code == "ABS_X":
                self.JOYSTICK_LEFT_X = self._clean_joystick(event.state)
            elif event.code == "ABS_RY":
                self.JOYSTICK_RIGHT_Y = self._clean_joystick(event.state)
            elif event.code == "ABS_RX":
                self.JOYSTICK_RIGHT_X = self._clean_joystick(event.state)
            elif event.code == "ABS_Z":
                self.TRIGGER_LEFT = self._clean_trigger(event.state)
            elif event.code == "ABS_RZ":
                self.TRIGGER_RIGHT = self._clean_trigger(event.state)
            elif event.code == "BTN_TL":
                self.btns.LB = ButtonState.PRESSED if event.state else ButtonState.RELEASED
            elif event.code == "BTN_TR":
                self.btns.RB = ButtonState.PRESSED if event.state else ButtonState.RELEASED
            elif event.code == "BTN_NORTH":
                self.btns.X = ButtonState.PRESSED if event.state else ButtonState.RELEASED
            elif event.code == "BTN_SOUTH":
                self.btns.A = ButtonState.PRESSED if event.state else ButtonState.RELEASED
            elif event.code == "BTN_WEST":
                self.btns.Y = ButtonState.PRESSED if event.state else ButtonState.RELEASED
            elif event.code == "BTN_EAST":
                self.btns.B = ButtonState.PRESSED if event.state else ButtonState.RELEASED
            elif event.code == "BTN_MODE":
                self.btns.MODE = ButtonState.PRESSED if event.state else ButtonState.RELEASED
            elif event.code == "ABS_HAT0X":
                self.btns.LEFT = ButtonState.PRESSED if event.state < 0 else ButtonState.RELEASED
                self.btns.RIGHT = ButtonState.PRESSED if event.state > 0 else ButtonState.RELEASED
            elif event.code == "ABS_HAT0Y":
                self.btns.UP = ButtonState.PRESSED if event.state < 0 else ButtonState.RELEASED
                self.btns.DOWN = ButtonState.PRESSED if event.state > 0 else ButtonState.RELEASED

    def _deadband(self, value: float, threshold: float) -> float:
        return value if abs(value) > threshold else 0.0

    def _clean_joystick(self, value: int) -> float:
        return self._deadband(value / 32768, .1)

    def _clean_trigger(self, value: int) -> float:
        return value / 255.0
