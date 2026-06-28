#!/usr/bin/env python3
"""
Power Board Communication Module

Handles CAN communication with the kbot power board using the specified protocol.
The power board sits on a dedicated CAN bus and controls power to each limb.
"""

import can
import struct
import time
import argparse
import signal
import sys
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum


class PowerBoardMessageType(IntEnum):
    """Power board CAN message types"""
    CONTROL = 0x1001   # Control frame (kbot to power board)
    QUERY = 0x1002     # Query frame (kbot to power board)  
    STATUS = 0x1003    # Status frame (power board to kbot)
    POWER_DATA = 0x1004  # Power consumption data (power board to kbot)


class PowerBoardFaultBits(IntEnum):
    """Fault status bit definitions"""
    POWER_CHIP_OVERCURRENT = 0
    POWER_CHIP_OVERTEMP = 1
    POWER_CHIP_SHORT_CIRCUIT = 2
    SAMPLING_OVERCURRENT = 3
    VBUS_OVERVOLTAGE = 4
    VBUS_UNDERVOLTAGE = 5
    VMBUS_OVERVOLTAGE = 6
    VMBUS_UNDERVOLTAGE = 7


@dataclass
class PowerBoardStatus:
    """Power board status data"""
    battery_voltage: float  # V
    motor_voltage: float    # V
    current_sampling: float # A
    fault_status: int      # Fault bits
    
    def get_faults(self) -> Dict[str, bool]:
        """Get individual fault status"""
        return {
            'power_chip_overcurrent': bool(self.fault_status & (1 << PowerBoardFaultBits.POWER_CHIP_OVERCURRENT)),
            'power_chip_overtemp': bool(self.fault_status & (1 << PowerBoardFaultBits.POWER_CHIP_OVERTEMP)),
            'power_chip_short_circuit': bool(self.fault_status & (1 << PowerBoardFaultBits.POWER_CHIP_SHORT_CIRCUIT)),
            'sampling_overcurrent': bool(self.fault_status & (1 << PowerBoardFaultBits.SAMPLING_OVERCURRENT)),
            'vbus_overvoltage': bool(self.fault_status & (1 << PowerBoardFaultBits.VBUS_OVERVOLTAGE)),
            'vbus_undervoltage': bool(self.fault_status & (1 << PowerBoardFaultBits.VBUS_UNDERVOLTAGE)),
            'vmbus_overvoltage': bool(self.fault_status & (1 << PowerBoardFaultBits.VMBUS_OVERVOLTAGE)),
            'vmbus_undervoltage': bool(self.fault_status & (1 << PowerBoardFaultBits.VMBUS_UNDERVOLTAGE)),
        }


@dataclass
class PowerBoardPowerData:
    """Power consumption data"""
    left_leg_power: float   # W
    right_leg_power: float  # W
    left_arm_power: float   # W
    right_arm_power: float  # W


@dataclass
class PowerBoardControl:
    """Power board control settings"""
    fan: bool = False
    precharge_voltage: bool = False
    output_to_motor: bool = False
    main_control: bool = False
    restart: bool = False
    clear_faults: bool = False
    auto_report: bool = False
    reserved: bool = False


class PowerBoard:
    """Power board communication interface"""
    
    POWER_BOARD_ID = 0xAA
    TIMEOUT = 1.0  # seconds
    
    def __init__(self, interface: str = "can3"):
        """Initialize power board communication
        
        Args:
            interface: CAN interface name (default: can3 - the 4th bus)
        """
        self.interface = interface
        self.bus = None
        
    def connect(self) -> bool:
        """Connect to CAN interface
        
        Returns:
            True if connection successful
        """
        try:
            self.bus = can.interface.Bus(
                channel=self.interface,
                interface='socketcan',
                bitrate=1000000  # 1 Mbps as specified
            )
            return True
        except Exception as e:
            print(f"Failed to connect to {self.interface}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from CAN interface"""
        if self.bus:
            self.bus.shutdown()
            self.bus = None
    
    def _create_can_id(self, message_type: PowerBoardMessageType) -> int:
        """Create 29-bit CAN ID
        
        Args:
            message_type: Type of message
            
        Returns:
            29-bit CAN ID
        """
        # 29-bit ID format:
        # Bits 0-7: Target Address (0xAA for power board)
        # Bits 8-20: Communication Type (message_type)
        # Bits 21-28: Reserved (0)
        can_id = (self.POWER_BOARD_ID & 0xFF) | ((message_type & 0x1FFF) << 8)
        return can_id | 0x80000000  # Set extended frame flag
    
    def send_control_frame(self, control: PowerBoardControl) -> bool:
        """Send control frame to power board
        
        Args:
            control: Control settings
            
        Returns:
            True if sent successfully
        """
        if not self.bus:
            return False
            
        can_id = self._create_can_id(PowerBoardMessageType.CONTROL)
        
        # Pack control data into 8 bytes
        data = [
            1 if control.fan else 0,
            1 if control.precharge_voltage else 0,
            1 if control.output_to_motor else 0,
            1 if control.main_control else 0,
            1 if control.restart else 0,
            1 if control.clear_faults else 0,
            1 if control.auto_report else 0,
            1 if control.reserved else 0,
        ]
        
        try:
            msg = can.Message(
                arbitration_id=can_id,
                data=data,
                is_extended_id=True
            )
            self.bus.send(msg)
            return True
        except Exception as e:
            print(f"Failed to send control frame: {e}")
            return False
    
    def send_query_frame(self) -> bool:
        """Send query frame to power board
        
        Returns:
            True if sent successfully
        """
        if not self.bus:
            return False
            
        can_id = self._create_can_id(PowerBoardMessageType.QUERY)
        
        try:
            msg = can.Message(
                arbitration_id=can_id,
                data=[0] * 8,  # Query frame uses 0 data
                is_extended_id=True
            )
            self.bus.send(msg)
            return True
        except Exception as e:
            print(f"Failed to send query frame: {e}")
            return False
    
    def _parse_status_frame(self, data: bytes) -> PowerBoardStatus:
        """Parse status frame data
        
        Args:
            data: 8-byte CAN data
            
        Returns:
            Parsed status data
        """
        if len(data) < 8:
            raise ValueError("Status frame must be 8 bytes")
            
        # Unpack data according to protocol
        battery_voltage_raw = struct.unpack('>H', data[0:2])[0]  # Big-endian uint16
        motor_voltage_raw = struct.unpack('>H', data[2:4])[0]
        current_sampling_raw = struct.unpack('>H', data[4:6])[0]
        fault_status = struct.unpack('>H', data[6:8])[0]
        
        # Convert raw values to actual units (0-65536 → 0-655.36)
        battery_voltage = battery_voltage_raw / 100.0
        motor_voltage = motor_voltage_raw / 100.0
        current_sampling = current_sampling_raw / 100.0
        
        return PowerBoardStatus(
            battery_voltage=battery_voltage,
            motor_voltage=motor_voltage,
            current_sampling=current_sampling,
            fault_status=fault_status
        )
    
    def _parse_power_data_frame(self, data: bytes) -> PowerBoardPowerData:
        """Parse power consumption frame data
        
        Args:
            data: 8-byte CAN data
            
        Returns:
            Parsed power data
        """
        if len(data) < 8:
            raise ValueError("Power data frame must be 8 bytes")
            
        # Unpack data according to protocol
        left_leg_power_raw = struct.unpack('>H', data[0:2])[0]
        right_leg_power_raw = struct.unpack('>H', data[2:4])[0] 
        left_arm_power_raw = struct.unpack('>H', data[4:6])[0]
        right_arm_power_raw = struct.unpack('>H', data[6:8])[0]
        
        # Convert raw values to actual units (0-65536 → 0-655.36)
        left_leg_power = left_leg_power_raw / 100.0
        right_leg_power = right_leg_power_raw / 100.0
        left_arm_power = left_arm_power_raw / 100.0
        right_arm_power = right_arm_power_raw / 100.0
        
        return PowerBoardPowerData(
            left_leg_power=left_leg_power,
            right_leg_power=right_leg_power,
            left_arm_power=left_arm_power,
            right_arm_power=right_arm_power
        )
    
    def get_status(self) -> Optional[Tuple[PowerBoardStatus, Optional[PowerBoardPowerData]]]:
        """Get power board status by sending query and waiting for response
        
        Returns:
            Tuple of (status, power_data) or None if failed
            power_data may be None if not available
        """
        if not self.bus:
            return None
            
        # Send query frame
        if not self.send_query_frame():
            return None
            
        # Wait for response
        status_data = None
        power_data = None
        start_time = time.time()
        
        while time.time() - start_time < self.TIMEOUT:
            try:
                msg = self.bus.recv(timeout=0.1)
                if msg is None:
                    continue
                    
                # Check if message is from power board
                if not msg.is_extended_id:
                    continue
                    
                # Extract message type and target address
                target_addr = msg.arbitration_id & 0xFF
                msg_type = (msg.arbitration_id >> 8) & 0x1FFF
                
                if target_addr != self.POWER_BOARD_ID:
                    continue
                    
                if msg_type == PowerBoardMessageType.STATUS:
                    status_data = self._parse_status_frame(msg.data)
                elif msg_type == PowerBoardMessageType.POWER_DATA:
                    power_data = self._parse_power_data_frame(msg.data)
                    
                # If we have status data, we can return (power data is optional)
                if status_data is not None:
                    return (status_data, power_data)
                    
            except Exception as e:
                print(f"Error receiving message: {e}")
                continue
                
        return None
    
    def enable_auto_report(self, enable: bool = True) -> bool:
        """Enable/disable auto-reporting (100ms intervals)
        
        Args:
            enable: True to enable auto-reporting
            
        Returns:
            True if command sent successfully
        """
        control = PowerBoardControl(auto_report=enable)
        return self.send_control_frame(control)
    
    def control_outputs(self, 
                       fan: bool = False,
                       precharge: bool = False,
                       motor_output: bool = False,
                       main_control: bool = False) -> bool:
        """Control power board outputs
        
        Args:
            fan: Enable cooling fan
            precharge: Enable precharge voltage
            motor_output: Enable output to motors
            main_control: Enable main control
            
        Returns:
            True if command sent successfully
        """
        control = PowerBoardControl(
            fan=fan,
            precharge_voltage=precharge,
            output_to_motor=motor_output,
            main_control=main_control
        )
        return self.send_control_frame(control)
    
    def clear_faults(self) -> bool:
        """Clear power board faults
        
        Returns:
            True if command sent successfully
        """
        control = PowerBoardControl(clear_faults=True)
        return self.send_control_frame(control)
    
    def restart(self) -> bool:
        """Restart power board
        
        Returns:
            True if command sent successfully
        """
        control = PowerBoardControl(restart=True)
        return self.send_control_frame(control)

    def stream_auto_report(self) -> None:
        """Stream auto-report data to stdout until interrupted"""
        if not self.bus:
            print("Error: Not connected to CAN bus")
            return
            
        print("Enabling auto-report mode...")
        if not self.enable_auto_report(True):
            print("Failed to enable auto-report")
            return
            
        print("Streaming power board data (Press Ctrl+C to stop)...")
        print("Timestamp        | Battery | Motor  | Current | Active Faults                | Left Leg | Right Leg | Left Arm | Right Arm")
        print("-" * 120)
        
        latest_status = None
        latest_power = None
        
        try:
            while True:
                try:
                    msg = self.bus.recv(timeout=0.5)
                    if msg is None:
                        continue
                        
                    if not msg.is_extended_id:
                        continue
                        
                    # Extract message type and target address
                    target_addr = msg.arbitration_id & 0xFF
                    msg_type = (msg.arbitration_id >> 8) & 0x1FFF
                    
                    if target_addr != self.POWER_BOARD_ID:
                        continue
                        
                    timestamp = time.strftime("%H:%M:%S.%f")[:-3]
                    
                    if msg_type == PowerBoardMessageType.STATUS:
                        latest_status = self._parse_status_frame(msg.data)
                        
                    elif msg_type == PowerBoardMessageType.POWER_DATA:
                        latest_power = self._parse_power_data_frame(msg.data)
                    
                    # Print combined data when we have both status and power data
                    if latest_status is not None:
                        faults = latest_status.get_faults()
                        active_faults = [name.replace('_', ' ').title() for name, active in faults.items() if active]
                        fault_str = ', '.join(active_faults) if active_faults else "None"
                        # Remove truncation - let it be as long as needed
                        
                        power_str = ""
                        if latest_power:
                            power_str = f"| {latest_power.left_leg_power:8.2f} | {latest_power.right_leg_power:9.2f} | {latest_power.left_arm_power:8.2f} | {latest_power.right_arm_power:9.2f}"
                        else:
                            power_str = f"| {'--':>8} | {'--':>9} | {'--':>8} | {'--':>9}"
                            
                        print(f"{timestamp} | {latest_status.battery_voltage:7.2f} | {latest_status.motor_voltage:6.2f} | {latest_status.current_sampling:7.2f} | {fault_str:<28} {power_str}")
                        
                except Exception as e:
                    print(f"Error receiving message: {e}")
                    continue
                    
        except KeyboardInterrupt:
            print("\nDisabling auto-report...")
            self.enable_auto_report(False)
            print("Stopped streaming.")

def print_status_once(power_board: PowerBoard) -> bool:
    """Print power board status once"""
    result = power_board.get_status()
    if result:
        status, power_data = result
        print(f"Battery Voltage: {status.battery_voltage:.2f}V")
        print(f"Motor Voltage: {status.motor_voltage:.2f}V") 
        print(f"Current: {status.current_sampling:.2f}A")
        print(f"Fault Status: 0x{status.fault_status:04x}")
        
        faults = status.get_faults()
        active_faults = [name for name, active in faults.items() if active]
        if active_faults:
            print(f"Active Faults: {', '.join(active_faults)}")
        else:
            print("No faults")
            
        if power_data:
            print(f"Left Leg Power: {power_data.left_leg_power:.2f}W")
            print(f"Right Leg Power: {power_data.right_leg_power:.2f}W")
            print(f"Left Arm Power: {power_data.left_arm_power:.2f}W")
            print(f"Right Arm Power: {power_data.right_arm_power:.2f}W")
        return True
    else:
        print("Failed to get status")
        return False


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='Power Board Communication Utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Get current status
  %(prog)s --auto-report                # Stream real-time data  
  %(prog)s --clear-faults               # Clear fault conditions
  %(prog)s --restart                    # Restart power board
  %(prog)s --control-outputs --fan --precharge  # Control specific outputs
  %(prog)s --interface can0             # Use different CAN interface
        """
    )
    
    parser.add_argument('--interface', '-i', default='can3',
                        help='CAN interface name (default: can3)')
    
    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument('--auto-report', '-a', action='store_true',
                              help='Enable auto-reporting and stream data to stdout')
    action_group.add_argument('--clear-faults', '-c', action='store_true',
                              help='Clear fault conditions')
    action_group.add_argument('--restart', '-r', action='store_true',
                              help='Restart power board')
    action_group.add_argument('--control-outputs', '-o', action='store_true',
                              help='Control power board outputs')
    
    # Control output options (only valid with --control-outputs)
    output_group = parser.add_argument_group('output control options')
    output_group.add_argument('--fan', action='store_true',
                              help='Enable cooling fan')
    output_group.add_argument('--precharge', action='store_true',
                              help='Enable precharge voltage')
    output_group.add_argument('--motor-output', action='store_true',
                              help='Enable output to motors')
    output_group.add_argument('--main-control', action='store_true',
                              help='Enable main control')
    
    args = parser.parse_args()
    
    # Validate arguments
    if any([args.fan, args.precharge, args.motor_output, args.main_control]) and not args.control_outputs:
        parser.error("Output control options (--fan, --precharge, --motor-output, --main-control) can only be used with --control-outputs")
    
    # Initialize power board
    power_board = PowerBoard(args.interface)
    
    if not power_board.connect():
        print(f"Failed to connect to power board on {args.interface}")
        return 1
    
    print(f"Connected to power board on {args.interface}")
    
    try:
        success = True
        
        if args.auto_report:
            power_board.stream_auto_report()
            
        elif args.clear_faults:
            print("Clearing faults...")
            if power_board.clear_faults():
                print("Fault clear command sent successfully")
                time.sleep(0.5)  # Brief delay before status check
                print_status_once(power_board)
            else:
                print("Failed to send fault clear command")
                success = False
                
        elif args.restart:
            print("Restarting power board...")
            if power_board.restart():
                print("Restart command sent successfully")
                print("Note: Power board will be unavailable during restart")
            else:
                print("Failed to send restart command")
                success = False
                
        elif args.control_outputs:
            print(f"Controlling outputs: fan={args.fan}, precharge={args.precharge}, motor_output={args.motor_output}, main_control={args.main_control}")
            if power_board.control_outputs(
                fan=args.fan,
                precharge=args.precharge,
                motor_output=args.motor_output,
                main_control=args.main_control
            ):
                print("Control command sent successfully")
                time.sleep(0.5)  # Brief delay before status check
                print_status_once(power_board)
            else:
                print("Failed to send control command")
                success = False
                
        else:
            # Default action: get status
            success = print_status_once(power_board)
            
    finally:
        power_board.disconnect()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
