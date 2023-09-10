from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import datetime
import math
import os
import psutil
import sys
import time
import rospy
import numpy as np


def getCircleTarget(pose, timestep, radius=0.075, freq=0.1):
    circ_target = pose[:]

    dx = math.cos((2 * math.pi * freq * timestep))
    dy = math.sin((2 * math.pi * freq * timestep))
    
    dL = np.sqrt(dx**2 + dy**2)
    
    circ_target[0] = pose[0] + radius * math.cos((2 * math.pi * freq * timestep))
    circ_target[1] = pose[1] + radius * math.sin((2 * math.pi * freq * timestep))

    # print("dL: ", dL)
    # print("pose: ", pose)
    # print("circ_target: ", circ_target)
    return circ_target


# Parameters
vel = 0.5
acc = 0.5
rtde_frequency = 125.0
dt = 1.0/rtde_frequency  # 8ms for 125Hz, 2ms 500Hz
# dt = 0.5
flags = RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT
# ur_cap_port = 50002
ur_cap_port = 30004
robot_ip = "10.0.0.1"

lookahead_time = 0.1
gain = 300

# ur_rtde realtime priorities
rt_receive_priority = 90
rt_control_priority = 85

rtde_r = RTDEReceive(robot_ip, rtde_frequency, [], True, False, rt_receive_priority)
rtde_c = RTDEControl(robot_ip, rtde_frequency, flags, ur_cap_port, rt_control_priority)

# Set application real-time priority
os_used = sys.platform
process = psutil.Process(os.getpid())
if os_used == "win32":  # Windows (either 32-bit or 64-bit)
    process.nice(psutil.REALTIME_PRIORITY_CLASS)
elif os_used == "linux":  # linux
    rt_app_priority = 80
    param = os.sched_param(rt_app_priority)
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
    except OSError:
        print("Failed to set real-time process scheduler to %u, priority %u" % (os.SCHED_FIFO, rt_app_priority))
    else:
        print("Process real-time priority set to: %u" % rt_app_priority)

time_counter = 0.0

# Move to init position using moveL
actual_tcp_pose = rtde_r.getActualTCPPose()
init_pose = getCircleTarget(actual_tcp_pose, time_counter)
rtde_c.moveL(init_pose, vel, acc)

previous_servo_target = 0.0

try:
    start_time = time.time()
    while True:

        start_time_i = time.time()

        t_start = rtde_c.initPeriod()
        servo_target = getCircleTarget(actual_tcp_pose, time_counter)
        rtde_c.servoL(servo_target, vel, acc, dt, lookahead_time, gain)
        rtde_c.waitPeriod(t_start)
        duration_i = time.time() - start_time_i
        # print("duration_i:", duration_i)
        time_counter += dt

        # servo_target = getCircleTarget(actual_tcp_pose, time_counter)
        # rtde_c.servoL(servo_target, vel, acc, dt, lookahead_time, gain)
        # rospy.sleep(dt)
        # time_counter += dt
        # duration_i = time.time() - start_time_i
        # print("duration_i:", duration_i)

        # print("dt: ", dt)
        # print(servo_target[0])
        target_step = servo_target[0] - previous_servo_target

        print("target_step: ", target_step)

        previous_servo_target = servo_target[0]


except KeyboardInterrupt:
    print("Control Interrupted!")
    rtde_c.servoStop()
    rtde_c.stopScript()
