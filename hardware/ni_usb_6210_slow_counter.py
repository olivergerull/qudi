# -*- coding: utf-8 -*-

"""
This file contains the Qudi hardware module to use a National Instruments USB 6210 card as slow counter.

Code has been derived  from "hardware/fpga_fastcounter/fast_counter_fpga_qo.py" and hardware/slow_counter_dummy.py
and was adapted to its new purpose by Oliver Gerull.

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

import time
import numpy as np
import statistics

from core.module import Base
from core.configoption import ConfigOption
from interface.slow_counter_interface import SlowCounterInterface
from interface.confocal_scanner_interface import ConfocalScannerInterface

from core.connector import Connector
import nidaqmx


class NIUSB6210SlowCounter(Base, SlowCounterInterface, ConfocalScannerInterface):
    """ Implementation of the SlowCounter interface methods for a NI USB 6210 Card.

    Example config for copy-paste:

    niusb_6210_slowcounter:
        module.Class: 'ni_usb_6210_slow_counter.NIUSB6210SlowCounter'
        gated: True
        counter_channels:
            - 'Dev1/ctr0'
            - 'Dev1/ctr1'
        pause_trigger_channels:
            - 'PFI1'
            - 'PFI2'
        pause_trigger_level: 'LOW'
        clock_frequency: 30  # in Hz, inverse measurement time when not given by GUI
        samples_to_read: 1   # number of samples for confocal scan (only median value will be kept to exclude outliers)
        connect:
            fitlogic: 'fitlogic'
    """
    # connectors (from confocal_scanner_dummy)
    fitlogic = Connector(interface='FitLogic')

    # config (from confocal_scanner_dummy)
    _clock_frequency = ConfigOption('clock_frequency', 50, missing='warn')

    # config options
    _gated = ConfigOption('gated', False, missing='warn')
    _counter_channels = ConfigOption(name='counter_channels', default=tuple(), missing='warn')
    _pause_trigger_channels = ConfigOption(name='pause_trigger_channels', default=tuple(), missing='info')
    _pause_trigger_level = ConfigOption(name='pause_trigger_level', default='LOW', missing='warn')
    _samples_to_read = ConfigOption(name='samples_to_read', default='1', missing='info')

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        # Internal parameters
        self._voltage_range = [0, 10]
        self._position_range = [[0, 60e-6], [0, 60e-6], [0, 60e-6], [0, 1e-6]]
        self._current_position = [0, 0, 0, 0][0:len(self.get_scanner_axes())]
        self._num_points = 500

        self.log.debug('The following configuration was found.')

        # checking for the right configuration
        for key in config.keys():
            self.log.info('{0}: {1}'.format(key, config[key]))

        # self._count_data = None         # actual count data
        self._opt_threshold = None     # Current optimization threshold in ODMR-Scan

        self.APD1 = nidaqmx.Task()
        self.APD2 = nidaqmx.Task()

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._fit_logic = self.fitlogic()

        # signal that device is running
        self.statusvar = 2

        # Current optimization threshold in ODMR-Scan
        self._opt_threshold = None

        # create and arm the APD counter channels of the NI card
        try:
            self.APD1.ci_channels.add_ci_count_edges_chan(self._counter_channels[0])
            self.APD2.ci_channels.add_ci_count_edges_chan(self._counter_channels[1])

            if self._gated:
                self.APD1.triggers.pause_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
                self.APD1.triggers.pause_trigger.dig_lvl_src = self._pause_trigger_channels[0]
                if self._pause_trigger_level == 'HIGH':
                    self.APD1.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.HIGH
                else:
                    self.APD1.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.LOW

                self.APD2.triggers.pause_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
                self.APD2.triggers.pause_trigger.dig_lvl_src = self._pause_trigger_channels[1]
                if self._pause_trigger_level == 'HIGH':
                    self.APD2.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.HIGH
                else:
                    self.APD2.triggers.pause_trigger.dig_lvl_when = nidaqmx.constants.Level.LOW

            # Start counting
            self.APD1.start()
            self.APD2.start()
        except:
            return -1

        return

    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        self.APD1.close()
        self.APD2.close()
        self._opt_threshold = None

        self.statusvar = -1
        return

    def get_constraints(self):
        """ Return a constraints class for the slow counter."""
        return None

    def set_up_clock(self, clock_frequency=None, clock_channel=None):
        """ Configures the hardware clock of the NiDAQ card to give the timing.

        @param float clock_frequency: if defined, this sets the frequency of the clock
        @param string clock_channel: if defined, this is the physical channel of the clock

        @return int: error code (0:OK, -1:error)
        """

        if clock_frequency is not None:
            self._clock_frequency = float(clock_frequency)

        self.log.debug('NIUSB6210SlowCounter>set_up_clock')
        return 0

    def set_up_counter(self,
                       counter_channels=None,
                       sources=None,
                       clock_channel=None,
                       counter_buffer=None):
        """ Configures the actual counter with a given clock.

        @param string counter_channels: if defined, this is the physical channel of the counter
        @param string sources: if defined, this is the physical channel where the photons are to count from
        @param string clock_channel: if defined, this specifies the clock for the counter
        @param string counter_buffer: if defined, this specifies the buffer of the counter

        @return int: error code (0:OK, -1:error)
        """
        return 0

    def get_counter(self, samples=None):
        """ Returns the current counts per second of the counter.

        @param int samples: if defined, number of samples to read in one go

        @return float: the photon counts per second
        """
        count_data = self.read_count_diff(1 / self._clock_frequency)

        # A new dimension has to be added. Otherwise the array won't be transposed in the next step.
        if count_data.ndim == 1:
            count_data = count_data[np.newaxis]
        # Transpose Array because Qudi needs this for slow counting
        count_data = count_data.transpose()

        return count_data

    def get_counter_channels(self):
        """ Returns the list of counter channel names.
        @return tuple(str): channel names
        Most methods calling this might just care about the number of channels, though.
        """
        return self.get_scanner_count_channels()

    def close_counter(self):
        """ Closes the counter and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        self.log.debug('NIUSB6210SlowCounter>close_counter')
        return 0

    def close_clock(self, power=0):
        """ Closes the clock and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        self.log.debug('NIUSB6210SlowCounter>close_clock')
        return 0

    @property
    def number_of_channels(self):
        """
        Read-only property to return the currently configured number of data channels of the confocal scan.

        @return int: the currently set number of channels
        """
        number = len(self.get_scanner_count_channels())
        return number

    @property
    def is_running(self):
        """
        Read-only flag indicating if the data acquisition is running.

        @return bool: Data acquisition is running (True) or not (False)
        """
        if self.statusvar == 2:
            return True
        else:
            return False

    def read_single_point(self):
        """
        This method will initiate a single sample read on each configured data channel.
        In general this sample may not be acquired simultaneous for all channels and timing in
        general can not be assured. Us this method if you want to have a non-timing-critical
        snapshot of your current data channel input.
        May not be available for all devices.
        The returned 1D numpy array will contain one sample for each channel.

        @return numpy.ndarray: 1D array containing one sample for each channel. Empty array
                               indicates error.
        """
        single_point = np.zeros(self.number_of_channels, dtype=np.float64)

        # Read the data from the counter
        single_point[0] = self.APD1.read()
        single_point[1] = self.APD2.read()
        single_point[2] = single_point[0] + single_point[1]     # The sum of both counters

        # This is the virtual channel for the Optimization threshold
        single_point[3] = self._opt_threshold

        return single_point

    def get_position_range(self):
        """ Returns the physical range of the scanner.

        @return float [4][2]: array of 4 ranges with an array containing lower
                              and upper limit
        """
        return self._position_range

    def set_position_range(self, myrange=None):
        """ Sets the physical range of the scanner.

        @param float [4][2] myrange: array of 4 ranges with an array containing
                                     lower and upper limit

        @return int: error code (0:OK, -1:error)
        """
        if myrange is None:
            myrange = [[0, 1e-6], [0, 1e-6], [0, 1e-6], [0, 1e-6]]

        if not isinstance(myrange, (frozenset, list, set, tuple, np.ndarray,)):
            self.log.error('Given range is no array type.')
            return -1

        if len(myrange) != 4:
            self.log.error('Given range should have dimension 4, but has '
                           '{0:d} instead.'.format(len(myrange)))
            return -1

        for pos in myrange:
            if len(pos) != 2:
                self.log.error('Given range limit {1:d} should have '
                               'dimension 2, but has {0:d} instead.'.format(len(pos), pos))
                return -1
            if pos[0] > pos[1]:
                self.log.error('Given range limit {0:d} has the wrong '
                               'order.'.format(pos))
                return -1

        self._position_range = myrange

        return 0

    def set_voltage_range(self, myrange=None):
        """ Sets the voltage range of the NI Card.

        @param float [2] myrange: array containing lower and upper limit

        @return int: error code (0:OK, -1:error)
        """
        return 0

    def get_scanner_axes(self):
        """ Cartesian axes from dummy scanner module.
        """
        return ['x', 'y', 'z', 'a']

    def get_scanner_count_channels(self):
        """ The counting channels. """
        scanner_count_channels = ['APD1', 'APD2', 'APDSum', 'Optim.-Threshold']
        return scanner_count_channels

    def set_up_scanner_clock(self, clock_frequency=None, clock_channel=None):
        """ Configures the hardware clock of the NiDAQ card to give the timing.

        @param float clock_frequency: if defined, this sets the frequency of the
                                      clock
        @param str clock_channel: if defined, this is the physical channel of
                                  the clock

        @return int: error code (0:OK, -1:error)
        """
        if clock_frequency is not None:
            self._clock_frequency = float(clock_frequency)

        self.log.debug('NIUSB6210SlowCounter>set_up_scanner_clock')
        return 0

    def set_up_scanner(self, counter_channels=None, sources=None,
                       clock_channel=None, scanner_ao_channels=None):
        """ Configures the actual scanner with a given clock.

        @param str counter_channels: if defined, this is the physical channel of
                                    the counter
        @param str sources: if defined, this is the physical channel where
                                  the photons are to count from
        @param str clock_channel: if defined, this specifies the clock for the
                                  counter
        @param str scanner_ao_channels: if defined, this specifies the analoque
                                        output channels

        @return int: error code (0:OK, -1:error)
        """
        return 0

    def scanner_set_position(self, x=None, y=None, z=None, a=None):
        """Move stage to x, y, z, a (where a is the fourth voltage channel).

        @param float x: position in x-direction (volts)
        @param float y: position in y-direction (volts)
        @param float z: position in z-direction (volts)
        @param float a: position in a-direction (volts)

        @return int: error code (0:OK, -1:error)
        """

        if self.module_state() == 'locked':
            self.log.error('A Scanner is already running, close this one first.')
            return -1

        self._current_position = [x, y, z, a][0:len(self.get_scanner_axes())]
        return 0

    def get_scanner_position(self):
        """ Get the current position of the scanner hardware.

        @return float[]: current position in (x, y, z, a).
        """
        return self._current_position[0:len(self.get_scanner_axes())]

    def _set_up_line(self, length=100):
        """ Sets up the analog output for scanning a line.

        @param int length: length of the line in pixel

        @return int: error code (0:OK, -1:error)
        """
        return 0

    def scan_line(self, line_path=None, pixel_clock=False):
        """ Scans a line and returns the counts on that line.

        @param float[][4] line_path: array of 4-part tuples defining the voltage points
        @param bool pixel_clock: whether we need to output a pixel clock for this line

        @return float[]: the photon counts per second
        """
        return 0

    def close_scanner(self):
        """ Closes the scanner and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        self.log.debug('NIUSB6210SlowCounter>close_scanner')
        return 0

    def close_scanner_clock(self, power=0):
        """ Closes the clock and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        self.log.debug('NIUSB6210SlowCounter>close_scanner_clock')
        return 0

    def reset_hardware(self):
        """
        Resets the NI hardware, so the connection is lost and other programs can access it.

        @return int: error code (0:OK, -1:error)
        """
        return 0

    # =================== Own Commands ========================
    def read_count_diff(self, integration_time):
        """
        This method will take the median of read_count_diff_once. So occasional too high or low values are thrown away.
        If you don't want this, just leave 'samples_to_read' at 1 in Config file.
        """
        single_point_diff = np.zeros(self.number_of_channels, dtype=np.float64)
        single_point_diff_s = np.zeros((self._samples_to_read, self.number_of_channels), dtype=np.float64)

        for sample in range(self._samples_to_read):
            single_point_diff_s[sample] = self.read_count_diff_once(integration_time)

        single_point_diff_ch = np.transpose(single_point_diff_s)

        single_point_diff[0] = statistics.median(single_point_diff_ch[0])
        single_point_diff[1] = statistics.median(single_point_diff_ch[1])

        # Third channel is the sum of both counters
        single_point_diff[2] = single_point_diff[0] + single_point_diff[1]

        # 4th channel is the Optimization Threshold
        single_point_diff[3] = self._opt_threshold

        return single_point_diff

    def read_count_diff_once(self, integration_time):
        """
        This method will read the counter, wait integration_time and read the counter again on
        each configured data channel. After that it calculates the difference between the counts
        and divide it by the integration time leading to the counts/s.

        The returned 1D numpy array will contain one counts/s value for each channel.

        @return numpy.ndarray: 1D array containing one counts/s value for each channel. Empty array
                               indicates error.
        """
        single_point_start = np.zeros(self.number_of_channels, dtype=np.float64)
        single_point_stop = np.zeros(self.number_of_channels, dtype=np.float64)

        # Read the data from the counter
        single_point_start[0] = self.APD1.read()
        single_point_start[1] = self.APD2.read()
        time_start = time.perf_counter_ns()  # perf_counter_ns() later leads to less rounding errors than perf_counter()

        # Wait the integration time
        time.sleep(integration_time)

        # Read the data from the counter
        single_point_stop[0] = self.APD1.read()
        single_point_stop[1] = self.APD2.read()
        time_stop = time.perf_counter_ns()

        # Actual time between both counts (sometimes differs strongly from integration_time)
        time_diff = (time_stop - time_start)

        # The actual counts are given by the difference
        single_point_diff = single_point_stop - single_point_start

        # Correct the count numbers by the actual measuring time
        single_point_diff = single_point_diff * (integration_time / time_diff) * 1e9

        # If one element of the count data is negative,
        # then the counter jumped to 0 in the counting process
        # and the counting should be repeated
        if np.min(single_point_diff) < 0:
            self.log.info('Counter jumped to 0. Repeating count.')
            return self.read_count_diff(integration_time)

        # counts/second = (counts/integration_time) * (1/integration_time)
        single_point_diff = single_point_diff / integration_time

        return single_point_diff
