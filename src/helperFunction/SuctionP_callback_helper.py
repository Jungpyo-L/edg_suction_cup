#!/usr/bin/env python
import numpy as np
import rospy
from suction_cup.msg import SensorPacket
from suction_cup.msg import cmdPacket
from scipy import signal
import threading

class P_CallbackHelp(object):
    def __init__(self):
        # Subscribe to SensorPacket
        self.sub = rospy.Subscriber("SensorPacket", SensorPacket, self.callback_P)

        # Command codes
        self.START_CMD = 2
        self.IDLE_CMD  = 3
        self.RECORD_CMD= 10

        # Create a command message and vacuum threshold
        self.msg2Sensor = cmdPacket()
        self.P_vac      = -10000.0

        # Publishers
        self.sensorCMD_Pub = rospy.Publisher('cmdPacket', cmdPacket, queue_size=10)
        self.callback_Pub  = rospy.Publisher('SensorCallback', SensorPacket, queue_size=10)
        self.callback_Pressure = SensorPacket()

        # Flag to indicate if publishing is still allowed
        self.publish_enabled = True

        # Number of sensors is initially unknown; we'll set it in callback.
        self.Psensor_Num = None
        self.BufferLen   = 7

        # Buffers to be initialized once we know Psensor_Num
        self.PressureBuffer       = None
        self.PressurePWMBuffer    = None
        self.PressureOffsetBuffer = None
        self.P_idx                = 0
        self.PWM_idx              = 0
        self.offset_idx           = 0

        # Pressure stats
        self.startPresAvg    = False
        self.startPresPWMAvg = False
        self.offsetMissing   = True
        self.thisPres        = None
        self.four_pressure   = None   # Will become an N-pressure vector
        self.four_pressurePWM= None   # Will also become an N-pressure vector
        self.PressureOffset  = None   # Will become an N-pressure vector
        self.power           = 0.0

        # For FFT
        self.samplingF       = 166
        self.FFTbuffer_size  = int(self.samplingF / 2)  # 166 is ~1 second
        self.lock            = threading.Lock()

    def initialize_arrays(self, num_ch):
        """
        (Re-)Initialize arrays/buffers for a new or changed number of chambers.
        """
        self.Psensor_Num = num_ch
        rospy.loginfo("Re-initializing arrays for %d chambers.", num_ch)

        # Basic ring buffers
        self.PressureBuffer = [[0.0] * self.Psensor_Num for _ in range(self.BufferLen)]
        self.P_idx          = 0
        self.startPresAvg   = False

        # For FFT computations
        self.PressurePWMBuffer    = np.zeros((self.FFTbuffer_size, self.Psensor_Num))
        self.PressureOffsetBuffer = np.zeros((51, self.Psensor_Num))
        self.four_pressurePWM     = np.zeros(self.Psensor_Num)
        self.PressureOffset       = np.zeros(self.Psensor_Num)
        self.thisPres             = np.zeros(self.Psensor_Num)
        self.PWM_idx              = 0
        self.startPresPWMAvg      = False

    def startSampling(self):
        self.msg2Sensor.cmdInput = self.START_CMD
        self.sensorCMD_Pub.publish(self.msg2Sensor)
    
    def stopSampling(self):
        self.msg2Sensor.cmdInput = self.IDLE_CMD
        self.sensorCMD_Pub.publish(self.msg2Sensor)

    def setNowAsOffset(self):
        """
        Safely copy PressureBuffer and compute offset.
        Prevents the 'RuntimeError: Inconsistent object...' error by acquiring self.lock.
        """
        if self.Psensor_Num is None:
            rospy.logwarn("Cannot set offset because number of chambers is unknown.")
            return

        self.PressureOffset *= 0
        rospy.sleep(0.5)

        # IMPORTANT: Acquire the lock before copying self.PressureBuffer
        with self.lock:
            buffer_copy = np.copy(self.PressureBuffer)

        # buffer_copy now has shape (BufferLen, Psensor_Num)
        self.PressureOffset = np.mean(buffer_copy, axis=0)
        rospy.loginfo("Pressure offset set to: %s", str(self.PressureOffset))

    def callback_P(self, data):
        """
        Called for each incoming SensorPacket message.
        data.ch:   number of chambers
        data.data: pressure array
        """
        # -- Step 0: If we're shutting down or have disabled publishing, return
        if not self.publish_enabled or rospy.is_shutdown():
            return

        # 1. If we haven't initialized arrays yet or the # of sensors changed, re-init
        if self.Psensor_Num is None or self.Psensor_Num != data.ch:
            self.initialize_arrays(data.ch)

        # 2. Fill in the ring buffer
        self.thisPres = np.array(data.data, dtype=float)

        with self.lock:
            self.PressureBuffer[self.P_idx] = self.thisPres - self.PressureOffset
            self.P_idx += 1
            if self.P_idx == len(self.PressureBuffer):
                self.startPresAvg = True
                self.P_idx = 0

        # 3. Fill in the PWM buffer for FFT computations
        self.PressurePWMBuffer[self.PWM_idx] = self.thisPres - self.PressureOffset
        self.PWM_idx += 1
        if self.PWM_idx == len(self.PressurePWMBuffer):
            self.startPresPWMAvg = True
            self.PWM_idx = 0

        # 4. If averaging flag is True, compute average across ring buffer
        if self.startPresAvg:
            buffer_np = np.array(self.PressureBuffer)
            averagePres_dummy = np.mean(buffer_np, axis=0)
            self.four_pressure = averagePres_dummy.tolist()

            # Publish callback, but only if still enabled
            if self.publish_enabled:
                self.callback_Pressure.ch   = self.Psensor_Num
                self.callback_Pressure.data = self.four_pressure
                self.callback_Pub.publish(self.callback_Pressure)

        # 5. If startPresPWMAvg is True, compute STFT for each sensor
        if self.startPresPWMAvg:
            # run stft on each pressure sensor
            averagePresPWM_dummy = np.zeros(self.Psensor_Num, dtype=float)
            fs   = self.samplingF
            N    = self.FFTbuffer_size
            fPWM = 30  # Hz

            for i in range(self.Psensor_Num):
                f, t, Zxx = signal.stft(self.PressurePWMBuffer[:, i], fs, nperseg=N)
                if len(f) > 1:
                    delta_f = f[1] - f[0]
                    idx = int(fPWM / delta_f) if delta_f != 0 else 0
                    idx = min(idx, len(f)-1)
                    power_spectrum = abs(Zxx[idx])
                    mean_power = np.mean(power_spectrum)
                    averagePresPWM_dummy[i] = mean_power

            self.four_pressurePWM = averagePresPWM_dummy

def shutdown_hook(p_helper):
    """
    This function is called by rospy.on_shutdown. It disables publishing
    to avoid 'publish() to a closed topic' errors.
    """
    rospy.loginfo("Shutdown hook called. Disabling publish, closing publishers.")
    
    # 1) Mark publish as disabled
    p_helper.publish_enabled = False
    
    # 2) Unregister publishers so they are truly closed
    p_helper.callback_Pub.unregister()
    p_helper.sensorCMD_Pub.unregister()
    
    # 3) Optionally, also unsubscribe from 'SensorPacket' if desired
    p_helper.sub.unregister()

def main():
    rospy.init_node("pressure_callback_node", anonymous=True)
    p_helper = P_CallbackHelp()

    # Register a shutdown hook
    rospy.on_shutdown(lambda: shutdown_hook(p_helper))

    # Example usage: Start sampling
    p_helper.startSampling()

    rospy.spin()

if __name__ == "__main__":
    main()
