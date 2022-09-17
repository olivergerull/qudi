# -*- coding: utf-8 -*-
"""
A module to control the Arduino Mega 2560 (by Oliver Gerull).
!!!!!! Please take a look at the README.md first. !!!!!!

Qudi is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Qudi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Qudi. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) the Qudi Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/Ulm-IQO/qudi/>
"""

from pymata4 import pymata4
from scipy.interpolate import interp1d
import numpy as np

from core.module import Base
from core.configoption import ConfigOption
from interface.motor_interface import MotorInterface

from interface.microwave_interface import MicrowaveInterface
from interface.microwave_interface import MicrowaveLimits
from interface.microwave_interface import MicrowaveMode
from interface.microwave_interface import TriggerEdge
import time
import ctypes


class Mega2560(Base, MotorInterface, MicrowaveInterface):
    """ Hardware module for Arduino Mega 2560.

    Example config for copy-paste:

    arduino_mega2560:
        module.Class: 'arduino.arduino_mega2560.Mega2560'
        pcf8591_i2c_address: 0x48       # I2C address of the PCF8591 module with byteshift: 0x90 >> 1 = 0x48
        trigger_pin: 22                 # Arduino pin that triggers SPI communication;
                                        # don't connect anything to this pin and the 20 pins after
        factor_pos_volt_x: 1.56e5       # conversion factor between position (in m) to voltage (in V)
        factor_pos_volt_y: 1.56e5       # conversion factor between position (in m) to voltage (in V)
        factor_pos_volt_z: 1.56e5       # conversion factor between position (in m) to voltage (in V)
        factor_volt_bit_x: 3.30e3       # conversion factor between voltage (in V) and bit value (0 - 65535)
        factor_volt_bit_y: 3.30e3       # conversion factor between voltage (in V) and bit value (0 - 65535)
        factor_volt_bit_z: 3.30e3       # conversion factor between voltage (in V) and bit value (0 - 65535)
        factor_volt_bit_f: 3.30e3       # conversion factor between voltage (in V) and bit value (0 - 65535)

        voltage_spi_x_min: 0            # minimum x output voltage of the DAC8734 in V
        voltage_spi_x_max: 10           # maximum x output voltage of the DAC8734 in V
        voltage_spi_y_min: 0            # minimum y output voltage of the DAC8734 in V
        voltage_spi_y_max: 10           # maximum y output voltage of the DAC8734 in V
        voltage_spi_z_min: 0            # minimum z output voltage of the DAC8734 in V
        voltage_spi_z_max: 10           # maximum z output voltage of the DAC8734 in V

        factor_volt_bit_a: 51           # conversion factor between PCF8591 output voltage (0-255) and bit value
        factor_frq_volt: 0.00001        # conversion factor between frequency (in Hz) and output voltage (0 - 65535)

        delay_i2c: 0.003                # time each i2c command needs until finished
        delay_spi: 0.007                # time each spi command needs until finished

        path_cal_vco: 'hardware\\arduino\\Calibration data\\VCOcal.txt' # path of calibration data for VCO
        path_cal_vca: 'hardware\\arduino\\Calibration data\\VCAcal.txt' # path of calibration data for VCA
    """

    # config options
    _pcf8591_i2c_address = ConfigOption(name='pcf8591_i2c_address', default=0x48, missing='nothing')
    _trigger_pin = ConfigOption(name='trigger_pin', default=22, missing='nothing')
    _factor_pos_volt_x = ConfigOption(name='factor_pos_volt_x', default=1.56e5, missing='warn', converter=float)
    _factor_pos_volt_y = ConfigOption(name='factor_pos_volt_y', default=1.56e5, missing='warn', converter=float)
    _factor_pos_volt_z = ConfigOption(name='factor_pos_volt_z', default=1.56e5, missing='warn', converter=float)
    _factor_volt_bit_x = ConfigOption(name='factor_volt_bit_x', default=3.30e3, missing='warn', converter=float)
    _factor_volt_bit_y = ConfigOption(name='factor_volt_bit_y', default=3.30e3, missing='warn', converter=float)
    _factor_volt_bit_z = ConfigOption(name='factor_volt_bit_z', default=3.30e3, missing='warn', converter=float)
    _factor_volt_bit_f = ConfigOption(name='factor_volt_bit_f', default=3.30e3, missing='warn', converter=float)
    _voltage_spi_x_min = ConfigOption(name='voltage_spi_x_min', default=0, missing='warn')
    _voltage_spi_x_max = ConfigOption(name='voltage_spi_x_max', default=10, missing='warn')
    _voltage_spi_y_min = ConfigOption(name='voltage_spi_y_min', default=0, missing='warn')
    _voltage_spi_y_max = ConfigOption(name='voltage_spi_y_max', default=10, missing='warn')
    _voltage_spi_z_min = ConfigOption(name='voltage_spi_z_min', default=0, missing='warn')
    _voltage_spi_z_max = ConfigOption(name='voltage_spi_z_max', default=10, missing='warn')
    _factor_volt_bit_a = ConfigOption(name='factor_volt_bit_a', default=-20, missing='nothing')
    _factor_frq_volt = ConfigOption(name='factor_frq_volt', default=0.00001, missing='nothing')
    _delay_i2c = ConfigOption(name='delay_i2c', default=0.003, missing='nothing')
    _delay_spi = ConfigOption(name='delay_spi', default=0.007, missing='nothing')
    _path_cal_vco = ConfigOption(name='path_cal_vco', default='', missing='warn')
    _path_cal_vca = ConfigOption(name='path_cal_vca', default='', missing='warn')

    # =================== Base Commands ========================
    def on_activate(self):
        # Set Windows timing resolution to 1 ms, otherwise time measurement fluctuates in the range of several 10 ms.
        ctypes.WinDLL('winmm').timeBeginPeriod(1)
        self.log.info('Windows Winmm timing resolution was set to 1 ms.')

        self.my_board = pymata4.Pymata4()

        # Initialisation of I2C communication
        self.my_board.set_pin_mode_i2c()

        # Control byte that activates the DAC
        self.my_board.i2c_write(self._pcf8591_i2c_address, [0x04])

        # Initialisation of SPI communication
        for pin in range(self._trigger_pin, self._trigger_pin + 21):
            self.my_board.set_pin_mode_digital_output(pin)  # make all pins output pins (input and pullup don't work)
            self.my_board.digital_write(pin, 0)  # set all pins to zero (just to be on the safe side)

        self.set_voltage_spi(self.my_board, 1, 0)
        self.set_voltage_spi(self.my_board, 2, 0)
        self.set_voltage_spi(self.my_board, 3, 0)

        # auxiliary variables
        self.x_old = 0
        self.y_old = 0
        self.z_old = 0

        self.mw_cw_power = -120.0
        self.mw_sweep_power = 0.0
        self.mw_cw_frequency = 2.87e9
        self.mw_frequency_list = list()
        self.mw_start_freq = 2.5e9
        self.mw_stop_freq = 3.1e9
        self.mw_step_freq = 2.0e6

        # frequency switching speed by a program in a list mode:
        self._FREQ_SWITCH_SPEED = 0.008  # Frequency switching speed in s

        self.current_output_mode = MicrowaveMode.CW  # Can be MicrowaveMode.CW, MicrowaveMode.LIST or
        # MicrowaveMode.SWEEP
        self.current_trig_pol = TriggerEdge.RISING  # Can be TriggerEdge.RISING or
        # TriggerEdge.FALLING
        self.output_active = False

        # Interpolation of VCO and VCA calibration values
        try:
            self.fct_vco = self.interpolate(self._path_cal_vco)
        except:
            self.log.error('There is a problem with the following file: %s', self._path_cal_vco)
            self.log.error("Please don't use microwave functions.")
            self.fct_vco = None

        try:
            self.fct_vca = self.interpolate(self._path_cal_vca)
        except:
            self.log.error('There is a problem with the following file: %s', self._path_cal_vca)
            self.log.error("Please don't use microwave functions.")
            self.fct_vca = None

        return

    def on_deactivate(self):
        pass

    # =================== MotorInterface Commands ========================
    def get_constraints(self):
        pass

    def move_rel(self, param_dict):
        pass

    def move_abs(self, param_dict):
        """Move an SPI controlled device to a specific position"""
        x_voltage = self._factor_pos_volt_x * param_dict['x']  # convert position to voltage
        y_voltage = self._factor_pos_volt_y * param_dict['y']  # convert position to voltage
        z_voltage = self._factor_pos_volt_z * param_dict['z']  # convert position to voltage

        x_new = int(self._factor_volt_bit_x * x_voltage)  # convert positions to bit value
        y_new = int(self._factor_volt_bit_y * y_voltage)  # convert positions to bit value
        z_new = int(self._factor_volt_bit_z * z_voltage)  # convert positions to bit value

        if x_new != self.x_old:
            if (x_voltage < self._voltage_spi_x_min) or (x_voltage > self._voltage_spi_x_max):
                raise Exception('SPI x voltage is not in range voltage_spi_x_min - voltage_spi_x_max.')
            else:
                self.set_voltage_spi(self.my_board, 1, x_new)  # set x-position
                self.x_old = x_new

        if y_new != self.y_old:
            if (y_voltage < self._voltage_spi_y_min) or (y_voltage > self._voltage_spi_y_max):
                raise Exception('SPI y voltage is not in range voltage_spi_y_min - voltage_spi_y_max.')
            else:
                self.set_voltage_spi(self.my_board, 2, y_new)  # set y-position
                self.y_old = y_new

        if z_new != self.z_old:
            if (z_voltage < self._voltage_spi_z_min) or (z_voltage > self._voltage_spi_z_max):
                raise Exception('SPI z voltage is not in range voltage_spi_z_min - voltage_spi_z_max.')
            else:
                self.set_voltage_spi(self.my_board, 3, z_new)  # set z-position
                self.z_old = z_new

        return 0

    def abort(self):
        pass

    def get_pos(self, param_list=None):
        pass

    def calibrate(self, param_list=None):
        pass

    def get_velocity(self, param_list=None):
        pass

    def set_velocity(self, param_dict):
        pass

    # =================== MicrowaveInterface Commands ========================
    def get_limits(self):
        """Microwave limits"""
        # ToDo: Could be calculated from hardware/arduino/Calibration data
        limits = MicrowaveLimits()
        limits.supported_modes = (MicrowaveMode.CW, MicrowaveMode.LIST, MicrowaveMode.SWEEP)

        # limits from the calibration data
        limits.min_frequency = 1.930750000000000188e+09
        limits.max_frequency = 2.960650000000000226e+09

        limits.min_power = -2.109993326733999197e+01
        limits.max_power = 4.139870882880003933e+00

        # other limits that are not so relevant but nevertheless asked for by Qudi
        limits.list_minstep = 0.001
        limits.list_maxstep = 20e9
        limits.list_maxentries = 10001

        limits.sweep_minstep = 0.001
        limits.sweep_maxstep = 20e9
        limits.sweep_maxentries = 10001
        return limits

    def get_status(self):
        """
        Gets the current status of the MW source, i.e. the mode (cw, list or sweep) and
        the output state (stopped, running)

        @return str, bool: mode ['cw', 'list', 'sweep'], is_running [True, False]
        """
        if self.current_output_mode == MicrowaveMode.CW:
            mode = 'cw'
        elif self.current_output_mode == MicrowaveMode.LIST:
            mode = 'list'
        elif self.current_output_mode == MicrowaveMode.SWEEP:
            mode = 'sweep'
        return mode, self.output_active

    def off(self):
        """ Switches off any microwave output.

        @return int: error code (0:OK, -1:error)
        """
        # # We don't want the mw to switched off here, as this would lead to unnecessary drifting
        # power_i2c = 255
        # self.set_voltage_i2c(self.my_board, power_i2c)  # set VCA attenuation
        # self.log.info('Microwave>off')

        # Signal that mw was switched off (required by Qudi, even though it's not true in our case)
        self.output_active = False
        return 0

    def get_power(self):
        """ Gets the microwave output power.

        @return float: the power set at the device in dBm
        """
        self.log.debug('Microwave>get_power')
        if self.current_output_mode == MicrowaveMode.CW:
            return self.mw_cw_power
        else:
            return self.mw_sweep_power

    def get_frequency(self):
        """
        Gets the frequency of the microwave output.
        Returns single float value if the device is in cw mode.
        Returns list if the device is in either list or sweep mode.

        @return [float, list]: frequency(s) currently set for this device in Hz
        """
        self.log.debug('Microwave>get_frequency')
        if self.current_output_mode == MicrowaveMode.CW:
            return self.mw_cw_frequency
        elif self.current_output_mode == MicrowaveMode.LIST:
            return self.mw_frequency_list
        elif self.current_output_mode == MicrowaveMode.SWEEP:
            return self.mw_start_freq, self.mw_stop_freq, self.mw_step_freq

    def cw_on(self):
        """
        Switches on cw microwave output.
        Must return AFTER the device is actually running.

        @return int: error code (0:OK, -1:error)
        """
        self.set_mw_frq(self.mw_cw_frequency)
        self.set_mw_pwr(self.mw_cw_power)

        self.current_output_mode = MicrowaveMode.CW

        self.log.info('Microwave>CW output on')
        self.output_active = True
        return 0

    def set_cw(self, frequency=None, power=None):
        """
        Configures the device for cw-mode and optionally sets frequency and/or power

        @param float frequency: frequency to set in Hz
        @param float power: power to set in dBm

        @return float, float, str: current frequency in Hz, current power in dBm, current mode

        Interleave option is used for arbitrary waveform generator devices.
        """
        self.log.debug('Microwave>set_cw, frequency: {0:f}, power {1:f}:'.format(frequency,
                                                                                      power))
        self.output_active = False
        self.current_output_mode = MicrowaveMode.CW
        if frequency is not None:
            self.mw_cw_frequency = frequency
        if power is not None:
            self.mw_cw_power = power
        return self.mw_cw_frequency, self.mw_cw_power, 'cw'

    def list_on(self):
        """
        Switches on the list mode microwave output.
        Must return AFTER the device is actually running.

        @return int: error code (0:OK, -1:error)
        """
        self.current_output_mode = MicrowaveMode.LIST
        time.sleep(1)
        self.output_active = True
        self.log.info('Microwave>List mode output on')
        return 0

    def set_list(self, frequency=None, power=None):
        """
        Configures the device for list-mode and optionally sets frequencies and/or power

        @param list frequency: list of frequencies in Hz
        @param float power: MW power of the frequency list in dBm

        @return list, float, str: current frequencies in Hz, current power in dBm, current mode
        """
        self.log.debug('Microwave>set_list, frequency_list: {0}, power: {1:f}'
                       ''.format(frequency, power))
        self.output_active = False
        self.current_output_mode = MicrowaveMode.LIST
        if frequency is not None:
            self.mw_frequency_list = frequency
        if power is not None:
            self.mw_cw_power = power
        return self.mw_frequency_list, self.mw_cw_power, 'list'

    def reset_listpos(self):
        """
        Reset of MW list mode position to start (first frequency step)

        @return int: error code (0:OK, -1:error)
        """
        return 0

    def sweep_on(self):
        """ Switches on the sweep mode.

        @return int: error code (0:OK, -1:error)
        """
        self.current_output_mode = MicrowaveMode.SWEEP
        self.output_active = True
        self.log.info('Microwave>Sweep mode output on')

        return 0

    def set_sweep(self, start=None, stop=None, step=None, power=None):
        """
        Configures the device for sweep-mode and optionally sets frequency start/stop/step
        and/or power

        @return float, float, float, float, str: current start frequency in Hz,
                                                 current stop frequency in Hz,
                                                 current frequency step in Hz,
                                                 current power in dBm,
                                                 current mode
        """
        self.log.debug('Microwave>set_sweep, start: {0:f}, stop: {1:f}, step: {2:f}, '
                       'power: {3:f}'.format(start, stop, step, power))
        self.output_active = False
        self.current_output_mode = MicrowaveMode.SWEEP
        if (start is not None) and (stop is not None) and (step is not None):
            self.mw_start_freq = start
            self.mw_stop_freq = stop
            self.mw_step_freq = step
        if power is not None:
            self.mw_sweep_power = power
        return self.mw_start_freq, self.mw_stop_freq, self.mw_step_freq, self.mw_sweep_power, 'sweep'

    def reset_sweeppos(self):
        """
        Reset of MW sweep mode position to start (start frequency)

        @return int: error code (0:OK, -1:error)
        """
        return 0

    def set_ext_trigger(self, pol, timing):
        """ Set the external trigger for this device with proper polarization.

        @param TriggerEdge pol: polarisation of the trigger (basically rising edge or falling edge)
        @param float timing: estimated time between triggers

        @return object: current trigger polarity [TriggerEdge.RISING, TriggerEdge.FALLING]
        """
        self.log.info('Microwave>ext_trigger set')
        self.current_trig_pol = pol
        return self.current_trig_pol, timing

    def trigger(self):
        """ Trigger the next element in the list or sweep mode programmatically.

        @return int: error code (0:OK, -1:error)

        Ensure that the Frequency was set AFTER the function returns, or give
        the function at least a save waiting time.
        """

        time.sleep(self._FREQ_SWITCH_SPEED)  # that is the switching speed
        return

    # =================== Own Commands ========================
    def set_mw_frq(self, frq):
        """ Set microwave frequency """
        if self.fct_vco is None:
            self.log.error('There was a problem with the VCO calibration.')
            return -1

        # Convert microwave frequency to voltage (using interpolation function)
        try:
            frq_voltage = self.fct_vco(frq * (10 ** -9))
        except ValueError:
            self.log.error('The frequency %f is not in the data range of %s.', frq * (10 ** (-9)), self._path_cal_vco)
            return -1

        # Convert voltage to bitcode
        frq_spi = int(self._factor_volt_bit_f * frq_voltage)

        # Set VCO frequency via SPI
        self.set_voltage_spi(self.my_board, 4, frq_spi)

    def set_mw_pwr(self, pwr):
        """ Set microwave power (attenuation in dBm) """
        if self.fct_vca is None:
            self.log.error('There was a problem with the VCA calibration.')
            return -1

        # Convert microwave attenuation (in dBm) to voltage (using interpolation function)
        try:
            att_voltage = self.fct_vca(pwr)
        except ValueError:
            self.log.error('The attenuation %f is not in the data range of %s.', pwr, self._path_cal_vca)
            return -1

        # Convert voltage to bitcode
        att_i2c = int(self._factor_volt_bit_a * att_voltage)

        # Set VCA attenuation vis I2C
        self.set_voltage_i2c(self.my_board, att_i2c)

    def set_voltage_spi(self, board, channel, voltage):
        """ Set voltage of the DAC8734 analog outputs (controlled by Arduino via SPI)

        Set voltage of one of the 4 analog outputs of the DAC8734 by SPI message from Arduino to DAC8734
        create and send a 20-bit SPI message according to DAC8734 Datasheet (first 4 bits: channel bit; 16 data bits)
        uses the 2ß digital Arduino pins after the trigger_pin for the SPI message and the trigger_pin as trigger
        don't connect anything to (trigger_pin) until (triggger_pin+20)

        Paramters:
        board:      Arduino-Board (initialized by self.my_board = pymata4.Pymata4())
        channel:    DAC analog output channel (allowed values: 1 - 4)
        voltage:    DAC analog output voltage (allowed values: 0 - 65535)
        """
        if (channel < 1) or (channel > 4):
            self.log.error('SPI channel is not in range 1 - 4.')
        elif (voltage < 0) or (voltage > 65535):
            self.log.error('Bitvalue of SPI voltage is not in range 0 - 65535.')
        else:
            channel = channel + 3  # channel register (channels 3-6 = 4 DAC analog outputs)
            channel = channel << 16  # leave 16 bits after the channel bits
            total_bits = channel + voltage  # add 16 voltage bits to channel bits
            bitcode = format(total_bits, '020b')  # make sure to have a 20 bit number

            for pin in range(self._trigger_pin + 1,
                             self._trigger_pin + 21):  # set 20 pins after the trigger_pin to 1 according to the bitcode
                if bitcode[pin - (self._trigger_pin + 1)] == '1':
                    board.digital_write(pin, 1)
                else:
                    board.digital_write(pin, 0)

            board.digital_write(self._trigger_pin, 1)  # set trigger_pin to 1, to make Arduino read the digital pins
            board.digital_write(self._trigger_pin, 0)  # set trigger_pin to 0, to stop Arduino from reading digital pins
            time.sleep(self._delay_spi)                # 'reaction time' of the DAC8734 we have to make Python wait,
                                                       # otherwise strange effects happen (e.g. wrong counts at the
                                                       # first pixel of each scan line)

    def set_voltage_i2c(self, board, voltage):
        """ Set voltage of the PCF8591 analog output (controlled by Arduino via I2C)

        Paramters:
        board:      Arduino-Board (initialized by self.my_board = pymata4.Pymata4())
        voltage:    PCF analog output voltage (allowed values: 0 - 255)
        """
        if (voltage < 0) or (voltage > 255):
            self.log.error('Voltage value for VCA is not in range 0-255.')
        else:
            device_address = 0x90 >> 1  # I2C address of the PCF8591 module (>> is a bitshift)
            board.i2c_write(device_address, [0x04])     # Control byte that activates the DAC
            board.i2c_write(device_address, [0x40, voltage])  # Set voltage
            time.sleep(self._delay_i2c)                 # 'reaction time' of the PCF8591 we have to make Python wait,
                                                        # otherwise strange effects happen (e.g. wrong counts at the
                                                        # first pixel of each scan line)

    def interpolate(self, path_calibr_values):
        """ Interpolate voltage values for VCA and VCO
        """
        # Read raw data from calibration files
        file_reader = open(path_calibr_values, "r")
        content_vco = file_reader.readlines()
        file_reader.close()

        # Split up raw data and save in arrays
        datapoints2_x = []
        datapoints2_y = []

        for i in content_vco:
            val_x, val_y = i.split(" ")
            val_x = float(val_x)
            val_y = float(val_y)
            datapoints2_x.append(val_x)
            datapoints2_y.append(val_y)

        datapoints_x = np.array(datapoints2_x)
        datapoints_y = np.array(datapoints2_y)

        # Interpolate the calibration data
        fct_interpolation = interp1d(datapoints_x, datapoints_y, kind='linear')
        return fct_interpolation
