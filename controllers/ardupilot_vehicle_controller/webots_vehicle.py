'''
This file implements a class that acts as a bridge between PX4 SITL and Webots

PX4_FLAKE8_CLEAN
'''

# Imports
import os
import sys
import time
import socket
import struct
import numpy as np
from threading import Thread
from typing import List, Union
from pymavlink import mavutil

# Here we set up environment variables so we can run this script
# as an external controller outside of Webots (useful for debugging)
# https://cyberbotics.com/doc/guide/running-extern-robot-controllers
if sys.platform.startswith("win"):
    WEBOTS_HOME = "C:\\Program Files\\Webots"
elif sys.platform.startswith("darwin"):
    WEBOTS_HOME = "/Applications/Webots.app"
elif sys.platform.startswith("linux"):
    WEBOTS_HOME = "/usr/local/webots"
else:
    raise Exception("Unsupported OS")

if os.environ.get("WEBOTS_HOME") is None:
    os.environ["WEBOTS_HOME"] = WEBOTS_HOME
else:
    WEBOTS_HOME = os.environ.get("WEBOTS_HOME")

os.environ["PYTHONIOENCODING"] = "UTF-8"
sys.path.append(f"{WEBOTS_HOME}/lib/controller/python")

DEBUG = False
HIL_SENSOR_UPDATE_RATE_HZ = 50   # 100Hz update rate for sensor data
HEARTBEAT_UPDATE_RATE_HZ = 1      # 1Hz update rate for heartbeat messages
MAIN_LOOP_SLEEP = 0.001           # 1ms sleep in main loop to prevent CPU spinning
ERROR_HANDLER_SLEEP = 0.1         # 100ms sleep when handling errors
CONNECTION_WAIT_SLEEP = 0.1       # 100ms sleep when waiting for connection
IMAGE_LOOP_SLEEP = 0.001          # 1ms sleep in image streaming loop

from controller import Robot, Camera, RangeFinder # noqa: E401, E402


class WebotsPX4Vehicle():
    """Class representing a PX4 controlled Webots Vehicle"""

    def __init__(self,
                 motor_names: List[str],
                 accel_name: str = "accelerometer",
                 imu_name: str = "inertial unit",
                 gyro_name: str = "gyro",
                 gps_name: str = "gps",
                 camera_name: str = None,
                 camera_fps: int = 10,
                 camera_stream_port: int = None,
                 rangefinder_name: str = None,
                 rangefinder_fps: int = 10,
                 rangefinder_stream_port: int = None,
                 instance: int = 0,
                 motor_velocity_cap: float = float('inf'),
                 reversed_motors: List[int] = None,
                 bidirectional_motors: bool = False,
                 uses_propellers: bool = True,
                 sitl_address: str = "127.0.0.1",
                 hil_sensor_rate: int = HIL_SENSOR_UPDATE_RATE_HZ,
                 heartbeat_rate: int = HEARTBEAT_UPDATE_RATE_HZ):
        """WebotsPX4Vehicle constructor

        Args:
            motor_names (List[str]): Motor names in PX4 numerical order (first motor is SERVO1 etc).
            accel_name (str, optional): Webots accelerometer name. Defaults to "accelerometer".
            imu_name (str, optional): Webots imu name. Defaults to "inertial unit".
            gyro_name (str, optional): Webots gyro name. Defaults to "gyro".
            gps_name (str, optional): Webots GPS name. Defaults to "gps".
            camera_name (str, optional): Webots camera name. Defaults to None.
            camera_fps (int, optional): Camera FPS. Lower FPS runs better in sim. Defaults to 10.
            camera_stream_port (int, optional): Port to stream grayscale camera images to.
                                                If no port is supplied the camera will not be streamed. Defaults to None.
            rangefinder_name (str, optional): Webots RangeFinder name. Defaults to None.
            rangefinder_fps (int, optional): RangeFinder FPS. Lower FPS runs better in sim. Defaults to 10.
            rangefinder_stream_port (int, optional): Port to stream rangefinder images to.
                                                     If no port is supplied the camera will not be streamed. Defaults to None.
            instance (int, optional): Vehicle instance number to match the SITL. This allows multiple vehicles. Defaults to 0.
            motor_velocity_cap (float, optional): Motor velocity cap. This is useful for the crazyflie
                                                  which default has way too much power. Defaults to float('inf').
            reversed_motors (list[int], optional): Reverse the motors (indexed from 1). Defaults to None.
            bidirectional_motors (bool, optional): Enable bidirectional motors. Defaults to False.
            uses_propellers (bool, optional): Whether the vehicle uses propellers.
                                              This is important as we need to linearize thrust if so. Defaults to True.
            sitl_address (str, optional): IP address of the SITL (useful with WSL2 eg \"172.24.220.98\").
                                          Defaults to "127.0.0.1".
            hil_sensor_rate (int, optional): Rate in Hz at which to send HIL sensor data. Defaults to global HIL_SENSOR_UPDATE_RATE_HZ.
            heartbeat_rate (int, optional): Rate in Hz at which to send heartbeat messages. Defaults to global HEARTBEAT_UPDATE_RATE_HZ.
        """
        # init class variables
        self.motor_velocity_cap = motor_velocity_cap
        self._instance = instance
        self._reversed_motors = reversed_motors
        self._bidirectional_motors = bidirectional_motors
        self._uses_propellers = uses_propellers
        self._webots_connected = True
        
        # Store update rates (override global defaults)
        self._hil_sensor_rate_hz = hil_sensor_rate
        self._heartbeat_rate_hz = heartbeat_rate

        print("Init devices")
        # setup Webots robot instance
        self.robot = Robot()

        # set robot time step relative to sim time step
        self._timestep = int(self.robot.getBasicTimeStep())

        # init sensors
        self.accel = self.robot.getDevice(accel_name)
        self.imu = self.robot.getDevice(imu_name)
        self.gyro = self.robot.getDevice(gyro_name)
        self.gps = self.robot.getDevice(gps_name)
        self.altimeter = self.robot.getDevice("altimeter")
        self.compass = self.robot.getDevice("compass")

        self.accel.enable(self._timestep)
        self.imu.enable(self._timestep)
        self.gyro.enable(self._timestep)
        self.gps.enable(self._timestep)
        self.altimeter.enable(self._timestep)
        self.compass.enable(self._timestep)

        # init camera
        if camera_name is not None:
            self.camera = self.robot.getDevice(camera_name)
            self.camera.enable(1000//camera_fps) # takes frame period in ms

            # start camera streaming thread if requested
            if camera_stream_port is not None:
                self._camera_thread = Thread(daemon=True,
                                             target=self._handle_image_stream,
                                             args=[self.camera, camera_stream_port])
                self._camera_thread.start()

        # init rangefinder
        if rangefinder_name is not None:
            self.rangefinder = self.robot.getDevice(rangefinder_name)
            self.rangefinder.enable(1000//rangefinder_fps) # takes frame period in ms

            # start rangefinder streaming thread if requested
            if rangefinder_stream_port is not None:
                self._rangefinder_thread = Thread(daemon=True,
                                                  target=self._handle_image_stream,
                                                  args=[self.rangefinder, rangefinder_stream_port])
                self._rangefinder_thread.start()

        print("Init motors")
        # init motors (and setup velocity control)
        self._motors = [self.robot.getDevice(n) for n in motor_names]
        for m in self._motors:
            m.setPosition(float('inf'))
            m.setVelocity(0)

        # Optionally skip simulation steps at startup (useful for letting sensors stabilize)
        for _ in range(100):
            self.robot.step(self._timestep)

        print("Start Mavlink thread")
        # start TCP MAVLink communication thread
        self._tcp_thread = Thread(daemon=True, target=self._handle_tcp_mavlink, args=[sitl_address, 4560])
        self._tcp_thread.start()
        
        # Initialize timestamp tracking for HIL messages
        import time
        self._start_time = time.time()
        self._last_hil_timestamp = 0


    # ------------------- END of init -----------------------    


    def _handle_tcp_mavlink(self, sitl_address: str = "127.0.0.1", tcp_port: int = 4560):
        """Handle TCP MAVLink communication with PX4 SITL
        
        Args:
            sitl_address (str): IP address of the SITL
            tcp_port (int): TCP port for MAVLink communication
        """
        print(f"Starting PX4 TCP MAVLink server on port {tcp_port} (I{self._instance})")
        
        # Create MAVLink connection as a TCP server
        connection_string = f'tcpin:0.0.0.0:{tcp_port}'
        print(f"PX4 TCP MAVLink server listening on port {tcp_port} (I{self._instance})")
        print(f"Update rates: HIL sensors at {self._hil_sensor_rate_hz}Hz, Heartbeat at {self._heartbeat_rate_hz}Hz (I{self._instance})")
        print(f"Timing parameters: Main loop: {MAIN_LOOP_SLEEP*1000:.1f}ms, Connection wait: {CONNECTION_WAIT_SLEEP*1000:.1f}ms, Error handler: {ERROR_HANDLER_SLEEP*1000:.1f}ms (I{self._instance})")
        self._mav_connection = mavutil.mavlink_connection(connection_string)

        msg = self._mav_connection.recv_match(blocking = True)
        if msg.get_type() != "COMMAND_LONG":
            print(f"Error: Expected COMMAND_LONG message, got {msg.get_type()} (I{self._instance})")
            return
        else:
            print(f"Received COMMAND_LONG message: {msg} (I{self._instance})")
        
        msg = self._mav_connection.recv_match(blocking = True)
        if msg.get_type() != "HEARTBEAT":
            print(f"Error: Expected HEARTBEAT message, got {msg.get_type()} (I{self._instance})")
            return
        else:
            print(f"Received HEARTBEAT message: {msg} (I{self._instance})")
        


        if not self._mav_connection.target_system:
            # No client connected yet, wait for messages
            msg = self._mav_connection.recv_match(blocking=False)
            if msg is not None:
                print(f"First MAVLink message received, client connected (I{self._instance})")

        try:
            # Initialize timing variables
            last_sensor_time = time.time()
            last_heartbeat_time = time.time()
            self._send_heartbeat_response()
            
            # # Main MAVLink communication loop
            while self.robot.step(self._timestep) != -1:
                msg = self._mav_connection.recv_match(blocking=False)
                if msg is not None:
                    if msg.get_type() == 'HIL_ACTUATOR_CONTROLS':
                        self._handle_hil_actuator_controls(msg)
                    elif msg.get_type() == 'HEARTBEAT':
                        self._send_heartbeat_response(msg)

                current_time = time.time()  
                if current_time - last_sensor_time >= 1.0 / self._hil_sensor_rate_hz:
                    self._send_hil_sensor_data()
                    last_sensor_time = current_time

                # periodic pings
                if current_time - last_heartbeat_time >= 1.0 / self._heartbeat_rate_hz:
                    self._send_heartbeat_response()
                    last_heartbeat_time = current_time
            
        except Exception as e:
            print(f"PX4 TCP MAVLink server error: {e} (I{self._instance})")
        finally:
            # Close MAVLink connection
            if hasattr(self, '_mav_connection'):
                self._mav_connection.close()
            print(f"PX4 TCP MAVLink server stopped (I{self._instance})")

    # _decode_mavlink_data method no longer needed since we're using pymavlink's built-in handling
    
    def _send_heartbeat_response(self):
        """Send heartbeat response"""
        try:
            if not hasattr(self, '_mav_connection'):
                return
                
            # Send heartbeat directly using the MAVLink connection
            self._mav_connection.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_QUADROTOR,  # type
                mavutil.mavlink.MAV_AUTOPILOT_PX4,  # autopilot - PX4 specific
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,  # base_mode
                0,  # custom_mode
                mavutil.mavlink.MAV_STATE_STANDBY  # system_status
            )
            
            if DEBUG:
                print(f"Sent PX4 heartbeat response (I{self._instance})")
            
        except Exception as e:
            print(f"Error sending heartbeat: {e}")
    
    def _send_param_response(self, msg):
        """Send parameter list response for PX4"""
        try:
            if not hasattr(self, '_mav_connection'):
                return
            
            # Send parameter value directly
            self._mav_connection.mav.param_value_send(
                b'SYS_AUTOSTART',  # param_id
                10016,  # param_value (iris model)
                mavutil.mavlink.MAV_PARAM_TYPE_INT32,  # param_type
                1,  # param_count
                0   # param_index
            )
            
            if DEBUG:
                print(f"Sent PX4 parameter response (I{self._instance})")
            
        except Exception as e:
            print(f"Error sending parameter response: {e}")
    
    def _handle_command_long(self, msg):
        """Handle MAVLink command long messages for PX4"""
        try:
            command = msg.command
            print(f"Received command: {command} (I{self._instance})")
            
            # Send command acknowledgment
            self._mav_connection.mav.command_ack_send(
                command,  # command
                mavutil.mavlink.MAV_RESULT_ACCEPTED  # result
            )
            
            if DEBUG:
                print(f"Sent command ACK for command {command} (I{self._instance})")
            
        except Exception as e:
            print(f"Error handling command: {e}")
    
    def _handle_hil_actuator_controls(self, msg):
        """Handle HIL_ACTUATOR_CONTROLS message from PX4"""
        try:
            # Extract motor controls from the message
            # controls is an array of 16 values, typically motors are first 4-8 values
            controls = msg.controls
            num_motors = min(len(self._motors), len(controls))
            
            # Extract motor commands (usually first 4 for quadcopter)
            motor_commands = controls[:num_motors]
            
            if DEBUG:
                print(f"PX4 Motor commands (I{self._instance}): {motor_commands[:num_motors]}")
            
            # Process motor commands similar to _handle_controls
            self._apply_motor_commands(motor_commands)
            
        except Exception as e:
            print(f"Error handling HIL actuator controls: {e}")
    
    def _apply_motor_commands(self, command_motors):
        """Apply motor commands to Webots motors
        
        Args:
            command_motors: List of motor control values (-1.0 to 1.0 for PX4)
        """
        try:
            # Motor mapping: PX4 SITL motor index -> Webots motor index
            # Based on your description:
            # PX4 Motor 1 (index 0) -> Webots Motor 4 (index 3)
            # PX4 Motor 2 (index 1) -> Webots Motor 3 (index 2) 
            # PX4 Motor 3 (index 2) -> Webots Motor 2 (index 1)
            # PX4 Motor 4 (index 3) -> Webots Motor 1 (index 0)
            px4_to_webots_mapping = [3, 2, 1, 0]  # PX4 index -> Webots index
            
            # For PX4 HIL_ACTUATOR_CONTROLS, the range is typically 0.0 to 1.0 for motors
            # 0.0 = motors off, 1.0 = full throttle
            if self._bidirectional_motors:
                # For bidirectional motors (rovers), might use -1 to 1 range
                scaled_commands = list(command_motors)
            else:
                # For unidirectional motors (quadcopters), use commands directly as they're 0 to 1
                scaled_commands = [max(0.0, min(1.0, cmd)) for cmd in command_motors]
            
            if DEBUG:
                print(f"PX4 motor commands (I{self._instance}): {scaled_commands}")
            
            # Linearize propeller thrust for `MOT_THST_EXPO=0`
            if self._uses_propellers:
                # `Thrust = thrust_constant * |omega| * omega` (ref https://cyberbotics.com/doc/reference/propeller)
                # if we set `omega = sqrt(input_throttle)` then `Thrust = thrust_constant * input_throttle`
                linearized_motor_commands = [np.sqrt(np.abs(v)) * np.sign(v) if v > 0 else 0 for v in scaled_commands]
            else:
                linearized_motor_commands = scaled_commands

            # Reverse motors if desired
            if self._reversed_motors:
                for m in self._reversed_motors:
                    if m-1 < len(linearized_motor_commands):
                        linearized_motor_commands[m-1] *= -1

            # Apply motor mapping and set velocities in Webots
            for px4_index, px4_command in enumerate(linearized_motor_commands):
                if px4_index < len(px4_to_webots_mapping) and px4_index < len(self._motors):
                    webots_index = px4_to_webots_mapping[px4_index]
                    if webots_index < len(self._motors):
                        velocity = px4_command * min(self._motors[webots_index].getMaxVelocity(), self.motor_velocity_cap)
                        self._motors[webots_index].setVelocity(velocity)
                        if DEBUG:
                            print(f"PX4 Motor {px4_index+1} -> Webots Motor {webots_index+1}: "
                                f"command={command_motors[px4_index]:.3f}, "
                                f"scaled={scaled_commands[px4_index]:.3f}, "
                                f"linearized={px4_command:.3f}, "
                                f"velocity={velocity:.3f}")
                    
        except Exception as e:
            print(f"Error applying motor commands: {e}")
        

    def _handle_controls(self, command: tuple):
        """Set the motor speeds based on the SITL command

        Args:
            command (tuple): tuple of motor speeds 0.0-1.0 where -1.0 is unused
        """

        # get only the number of motors we have
        command_motors = command[:len(self._motors)]
        if -1 in command_motors:
            print(f"Warning: SITL provided {command.index(-1)} motors "
                  f"but model specifies {len(self._motors)} (I{self._instance})")

        # print motor values
        if DEBUG:
            print(f"Motor commands (I{self._instance}): {command_motors}")

        # scale commands to -1.0-1.0 if the motors are bidirectional (ex rover wheels)
        if self._bidirectional_motors:
            command_motors = [v*2-1 for v in command_motors]

        # linearize propeller thrust for `MOT_THST_EXPO=0`
        if self._uses_propellers:
            # `Thrust = thrust_constant * |omega| * omega` (ref https://cyberbotics.com/doc/reference/propeller)
            # if we set `omega = sqrt(input_thottle)` then `Thrust = thrust_constant * input_thottle`
            linearized_motor_commands = [np.sqrt(np.abs(v))*np.sign(v) for v in command_motors]

        # reverse motors if desired
        if self._reversed_motors:
            for m in self._reversed_motors:
                linearized_motor_commands[m-1] *= -1

        # set velocities of the motors in Webots
        for i, m in enumerate(self._motors):
            m.setVelocity(linearized_motor_commands[i] * min(m.getMaxVelocity(), self.motor_velocity_cap))

    def webots_connected(self) -> bool:
        """Check if Webots client is connected"""
        return self._webots_connected

    def _send_hil_sensor_data(self):
        """Send HIL sensor data to PX4 via MAVLink"""
        try:
            if not hasattr(self, '_mav_connection'):
                return
                
            # Get sensor data from Webots
            i = self.imu.getRollPitchYaw()
            g = self.gyro.getValues()
            a = self.accel.getValues()
            gps_pos = self.gps.getValues()
            gps_vel = self.gps.getSpeedVector()
            
            # Get altimeter and compass data
            altitude_m = self.altimeter.getValue()  # meters above sea level
            compass_values = self.compass.getValues()  # north direction vector
            compass_values = np.round(np.array(compass_values), 3)
            
            if DEBUG:
                print(f"GPS position (I{self._instance}): {gps_pos}, velocity: {gps_vel}")

            a = np.round(np.array(a), 5)
            g = np.round(np.array(g), 5)

            # Convert to NED coordinate system (PX4 standard)
            # Webots uses ENU, PX4 uses NED
            # Accelerations in m/s²
            # Add small noise to match test.py
            xacc = a[0]  # ENU X → NED X
            yacc = -a[1]  # ENU -Y → NED Y
            zacc = -a[2]  # ENU -Z → NED Z


            xgyro = g[0]  # ENU X → NED X
            ygyro = -g[1]  # ENU -Y → NED Y
            zgyro = -g[2]  # ENU -Z → NED Z
            
            north_frd_x = compass_values[1]   # ENU Y (North) → FRD X (Forward)
            north_frd_y = compass_values[0]   # ENU X (East) → FRD Y (Right)
            north_frd_z = -compass_values[2]  # ENU Z (Up) → FRD Z (Down)

            EARTH_MAG_STRENGTH = 0.5  # Gauss
            mag_norm = np.sqrt(north_frd_x**2 + north_frd_y**2 + north_frd_z**2)

            scale_factor = EARTH_MAG_STRENGTH / mag_norm
            xmag = compass_values[0] * EARTH_MAG_STRENGTH    # X unchanged
            ymag = -compass_values[1] * EARTH_MAG_STRENGTH   # Y negated  
            zmag = -compass_values[2] * EARTH_MAG_STRENGTH   # Z negated      
            # print mag values
            if DEBUG:
                print(f"Magnetometer (I{self._instance}): "
                        f"xmag={xmag:.3f}, ymag={ymag:.3f}, zmag={zmag:.3f} (normalized to {EARTH_MAG_STRENGTH} Gauss)")
            # Calculate barometric pressure from altitude
            # Standard atmospheric pressure at sea level is 1013.25 hPa
            # But use a slightly different value based on test.py
            abs_pressure = self.calculate_pressure_from_altitude(altitude_m)
            diff_pressure = 0.0  # No airspeed measurement, set to 0
            
            # Pressure altitude derived from barometric pressure
            # Using altitude directly from the altimeter
            pressure_alt = altitude_m  # meters with noise
            
            # Create proper timestamp for HIL_SENSOR message
            # Use system time to ensure monotonic increasing timestamps
            current_time = time.time()
            time_usec = int((current_time - self._start_time) * 1000000)  # microseconds since start
            
            # Ensure timestamp is always increasing
            if time_usec <= self._last_hil_timestamp:
                time_usec = self._last_hil_timestamp + 1000  # Add 1ms
            self._last_hil_timestamp = time_usec
            
            # Temperature in Celsius, typically room temperature
            temperature = 25.0  # °C
            
            # Fields updated bitmask (all sensors)
            # 0x1FFF means all fields are valid (bits 0-12 set)
            fields_updated = 0x1FFF
            
            if DEBUG:
                print(f"Sending HIL sensor data (I{self._instance}): "
                    f"xacc={xacc:.5f}, yacc={yacc:.5f}, zacc={zacc:.5f}, "
                    f"xgyro={xgyro:.5f}, ygyro={ygyro:.5f}, zgyro={zgyro:.5f}, "
                    f"xmag={xmag:.3f}, ymag={ymag:.3f}, zmag={zmag:.3f}, "
                    f"abs_pressure={abs_pressure:.2f}hPa, diff_pressure={diff_pressure:.2f}hPa, "
                    f"pressure_alt={pressure_alt:.2f}m, temperature={temperature:.1f}°C, "
                    f"time_usec={time_usec} (I{self._instance})")
            


            # Send HIL_SENSOR message directly using the MAVLink connection
            # Format matches the test.py reference implementation
            self._mav_connection.mav.hil_sensor_send(
                 time_usec           = time_usec        ,
                xacc                = xacc              ,
                yacc                = yacc              ,
                zacc                = zacc              ,
                xgyro               = xgyro             ,
                ygyro               = ygyro             ,
                zgyro               = zgyro             ,
                xmag                = xmag              ,
                ymag                = ymag              ,
                zmag                = zmag              ,
                abs_pressure        = abs_pressure      ,
                diff_pressure       = diff_pressure     ,
                pressure_alt        = pressure_alt      ,
                temperature         = temperature       ,
                fields_updated      = fields_updated    ,
            )
            
            # Also send HIL_GPS if GPS data available
            if gps_pos[0] != 0 or gps_pos[1] != 0:
                self._send_hil_gps_data(gps_pos, gps_vel)
                
        except Exception as e:
            print(f"Error sending HIL sensor data: {e}")

    def _send_hil_gps_data(self, gps_pos, gps_vel):
        """Send HIL GPS data to PX4"""
        try:
            if not hasattr(self, '_mav_connection'):
                return
                
            # Use same timestamp system as HIL_SENSOR
            current_time = time.time()
            time_usec = int((current_time - self._start_time) * 1000000)  # microseconds since start
            
            # Round extremely small values to zero (fixes floating point noise issues)
            # Convert list to numpy array for easier manipulation
            gps_vel = np.array(gps_vel)
        
            # Round to 3 decimal places to eliminate floating point noise
            gps_vel = np.round(gps_vel, 3)
        
            # Values smaller than 0.001 are effectively zero
            gps_vel[np.abs(gps_vel) < 0.001] = 0.0

            # Check for NaN values in the input data
                    
            # Convert position based on data format
            # - If data appears to be in degrees already, use as is
            # - If data appears to be in local coordinates, convert from meters to lat/lon
            
            lat = int(gps_pos[0] * 1e7)
            lon = int(-gps_pos[1] * 1e7)
            
            # Convert altitude to millimeters (positive up)
            alt = int(-gps_pos[2] * 1000)  # Convert meters to millimeters
            
            # Convert velocities from Webots (ENU) to NED coordinate system
            # Ensure values are within int16 range (-32768 to 32767)
            def clamp_int16(val):
                if np.isnan(val) or val is None:
                    return 0
                return max(-32767, min(32767, int(val)))
            
            # Scale velocity to cm/s and convert ENU to NED
            vn = clamp_int16(-gps_vel[1] * 100)    # Y in ENU → North in NED
            ve = clamp_int16(gps_vel[0] * 100)    # X in ENU → East in NED
            vd = clamp_int16(-gps_vel[2] * 100)   # -Z in ENU → Down in NED
            
            # Calculate ground speed (2D horizontal velocity)
            vel = int(np.sqrt(vn**2 + ve**2))  # cm/s
            
            # Calculate course over ground (heading) in centidegrees
            # atan2(ve, vn) gives angle in radians from North
            if abs(vn) < 1e-6 and abs(ve) < 1e-6:
                cog = 0  # No movement, default to North
            else:
                heading_rad = np.arctan2(ve, vn)  # Heading in radians
                heading_deg = np.degrees(heading_rad)  # Convert to degrees
                if heading_deg < 0:
                    heading_deg += 360  # Convert to 0-360 range
                cog = int(heading_deg * 100)  # Convert to centidegrees
            
            eph = 30  # Horizontal position accuracy in cm
            epv = 40  # Vertical position accuracy in cm
            
            # Print debug information
            if DEBUG:
                print(f"Sending HIL GPS data (I{self._instance}): "
                    f"lat={lat/1e7:.6f}°, lon={lon/1e7:.6f}°, alt={alt/1000:.1f}m, "
                    f"vel={vel}cm/s, vn={vn}cm/s, ve={ve}cm/s, vd={vd}cm/s, cog={cog/100:.1f}°")

            # import inspect
            # print(inspect.signature(self._mav_connection.mav.hil_gps_send))
            # Send HIL_GPS message with proper parameters
            self._mav_connection.mav.hil_gps_send(
                time_usec,             # Timestamp (microseconds)
                3,                     # fix_type (3D fix)
                lat,                   # Latitude (WGS84) [degE7]
                lon,                   # Longitude (WGS84) [degE7]
                alt,                   # Altitude (MSL) [mm]
                eph,                   # GPS HDOP horizontal dilution of position [cm]
                epv,                   # GPS VDOP vertical dilution of position [cm]
                vel,                   # GPS ground speed [cm/s]
                vn,                    # North velocity [cm/s]
                ve,                    # East velocity [cm/s]
                vd,                    # Down velocity [cm/s]
                cog,                   # Course over ground [cdeg]
                10,                    # Number of satellites visible
            )
            
            if DEBUG:
                print(f"HIL GPS sent: lat={lat/1e7:.6f}°, lon={lon/1e7:.6f}°, alt={alt/1000:.1f}m, vel={vel}cm/s, cog={cog/100:.1f}°")
            
        except Exception as e:
            print(f"Error sending HIL GPS data: {e}")
            if DEBUG:
                print(f"GPS position: {gps_pos}, velocity: {gps_vel}")


    # --------------- Unused for now ---------------

    def _handle_image_stream(self, camera: Union[Camera, RangeFinder], port: int):
        """Stream grayscale images over TCP

        Args:
            camera (Camera or RangeFinder): the camera to get images from
            port (int): port to send images over
        """

        # get camera info
        # https://cyberbotics.com/doc/reference/camera
        if isinstance(camera, Camera):
            cam_sample_period = self.camera.getSamplingPeriod()
            cam_width = self.camera.getWidth()
            cam_height = self.camera.getHeight()
            print(f"Camera stream started at 127.0.0.1:{port} (I{self._instance}) "
                  f"({cam_width}x{cam_height} @ {1000/cam_sample_period:0.2f}fps)")
        elif isinstance(camera, RangeFinder):
            cam_sample_period = self.rangefinder.getSamplingPeriod()
            cam_width = self.rangefinder.getWidth()
            cam_height = self.rangefinder.getHeight()
            print(f"RangeFinder stream started at 127.0.0.1:{port} (I{self._instance}) "
                  f"({cam_width}x{cam_height} @ {1000/cam_sample_period:0.2f}fps)")
        else:
            print(sys.stderr, f"Error: camera passed to _handle_image_stream is of invalid type "
                              f"'{type(camera)}' (I{self._instance})")
            return

        # create a local TCP socket server
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', port))
        server.listen(1)

        # continuously send images
        while self._webots_connected:
            # wait for incoming connection
            conn, _ = server.accept()
            print(f"Connected to camera client (I{self._instance})")

            # send images to client
            try:
                while self._webots_connected:
                    # delay at sample rate
                    start_time = self.robot.getTime()

                    # get image
                    if isinstance(camera, Camera):
                        img = self.get_camera_gray_image()
                    elif isinstance(camera, RangeFinder):
                        img = self.get_rangefinder_image()

                    if img is None:
                        print(f"No image received (I{self._instance})")
                        time.sleep(cam_sample_period/1000)  # Use the camera sample period directly here as it's already calculated
                        continue

                    # create a header struct with image size
                    header = struct.pack("=HH", cam_width, cam_height)

                    # pack header and image and send
                    data = header + img.tobytes()
                    conn.sendall(data)

                    # delay at sample rate
                    while self.robot.getTime() - start_time < cam_sample_period/1000:
                        time.sleep(IMAGE_LOOP_SLEEP)

            except ConnectionResetError:
                pass
            except BrokenPipeError:
                pass
            finally:
                conn.close()
                print(f"Camera client disconnected (I{self._instance})")

    def get_camera_gray_image(self) -> np.ndarray:
        """Get the grayscale image from the camera as a numpy array of bytes"""
        img = self.get_camera_image()
        img_gray = np.average(img, axis=2).astype(np.uint8)
        return img_gray

    def get_camera_image(self) -> np.ndarray:
        """Get the RGB image from the camera as a numpy array of bytes"""
        img = self.camera.getImage()
        img = np.frombuffer(img, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
        return img[:, :, :3] # RGB only, no Alpha

    def get_rangefinder_image(self, use_int16: bool = False) -> np.ndarray:
        """Get the rangefinder depth image as a numpy array of int8 or int16"""\

        # get range image size
        height = self.rangefinder.getHeight()
        width = self.rangefinder.getWidth()

        # get image, and convert raw ctypes array to numpy array
        # https://cyberbotics.com/doc/reference/rangefinder
        image_c_ptr = self.rangefinder.getRangeImage(data_type="buffer")
        img_arr = np.ctypeslib.as_array(image_c_ptr, (width*height,))
        img_floats = img_arr.reshape((height, width))

        # normalize and set unknown values to max range
        range_range = self.rangefinder.getMaxRange() - self.rangefinder.getMinRange()
        img_normalized = (img_floats - self.rangefinder.getMinRange()) / range_range
        img_normalized[img_normalized == float('inf')] = 1

        # convert to int8 or int16, allowing for the option of higher precision if desired
        if use_int16:
            img = (img_normalized * 65535).astype(np.uint16)
        else:
            img = (img_normalized * 255).astype(np.uint8)

        return img
    
    def calculate_pressure_from_altitude(self, altitude_m):
        """
        Calculate barometric pressure from altitude using the standard atmospheric model
        
        Args:
            altitude_m (float): Altitude in meters above sea level
            
        Returns:
            float: Pressure in hPa (hectopascals)
        """
        # Standard atmospheric constants
        P0 = 1013.25  # Sea level standard atmospheric pressure in hPa
        T0 = 288.15   # Sea level standard temperature in Kelvin (15°C)
        L = 0.0065    # Temperature lapse rate in K/m
        g = 9.80665   # Acceleration due to gravity in m/s²
        M = 0.0289644 # Molar mass of Earth's air in kg/mol
        R = 8.31432   # Universal gas constant in N⋅m/(mol⋅K)
        
        # Calculate pressure using the barometric formula
        # P = P0 * (1 - L*h/T0)^(g*M/(R*L))
        pressure_hpa = P0 * pow(1 - (L * altitude_m / T0), (g * M) / (R * L))
        
        return pressure_hpa