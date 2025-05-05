#!/usr/bin/env python

## ESP32-S3 feather pressure sensor reading and publish import rospy

import rospy
import numpy as np
import os, sys
from suction_cup.msg import SensorPacket
from suction_cup.msg import cmdPacket

# serial communication
import serial
from serial import Serial
import struct


IDLE = 0
STREAMING = 1

NO_CMD = 0
START_CMD = 2
IDLE_CMD = 3

currState = IDLE
CMD_in = NO_CMD


def callback(data):
    global CMD_in        
    CMD_in = data.cmdInput


def main(args):
    global currState
    global CMD_in

    rospy.init_node('ESP32_Pressure')
    
    #Sensor reading is published to topic 'SensorPacket'
    pub = rospy.Publisher('SensorPacket', SensorPacket, queue_size=10)
    rospy.Subscriber("cmdPacket",cmdPacket, callback)
    msg = SensorPacket()
    # depending on the number of channel (args.ch), the size of data will be different
    msg.ch = args.ch
    msg.data = [0.0]*args.ch
    # msg.data = [0.0, 0.0, 0.0, 0.0] 

    # ser = serial.Serial("/dev/ttyACM0", baudrate=115200, timeout=1, write_timeout=1)
    ser = serial.Serial("/dev/ttyPressure", baudrate=115200, timeout=1, write_timeout=1)
    ser.flushInput()
 
    while not rospy.is_shutdown():
        try:
            if currState == IDLE and CMD_in == START_CMD:
                CMD_in = NO_CMD
                ser.write(struct.pack('<B', ord("i")))
                rospy.sleep(0.01)
                ser.write(struct.pack('<B', ord("s")))
                rospy.sleep(0.01)
                while not CMD_in == IDLE_CMD and not rospy.is_shutdown():
                    ser_bytes = ser.readline().decode("utf-8")
                    split_data = ser_bytes.split(' ')                    
                    # rewrite above code depending on the number of channel
                    # Ensure we have enough split elements to match the desired number of chambers
                    if len(split_data) < args.ch:
                        rospy.logwarn("Received fewer data points than expected. Received %d, expected %d", 
                                    len(split_data), args.ch)
                        return  # Or handle it however is appropriate for your application

                    # Assign values dynamically
                    for i in range(args.ch):
                        msg.data[i] = float(split_data[i])
                    msg.header.stamp = rospy.Time.now()
                    pub.publish(msg)
                    print(msg)

                # ser.write("i" + "\r\n")
                ser.write(struct.pack('<B', ord("i")))
                ser.flushInput()
                rospy.sleep(0.01)            
                CMD_in = NO_CMD
                currState = IDLE
        
        except Exception as e:
            print("SensorComError: " + str(e))
            pass
    print("ESP32 Sensor Reading Done")

if __name__ == '__main__':
    try:
        print("Started!")
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--ch', type=int, help='number of channel', default= 4)

        args = parser.parse_args()    
        main(args)

    except rospy.ROSInterruptException: 
        print("oops")
        pass
