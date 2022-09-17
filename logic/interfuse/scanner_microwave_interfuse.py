"""
This file contains the Qudi interfuse between MicrowaveController and Piezo Positioner.
It was written by Oliver Gerull.

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

import numpy as np
import time
import random

from core.module import Base
from core.connector import Connector
from core.configoption import ConfigOption
from interface.odmr_counter_interface import ODMRCounterInterface


class ScannerMicrowaveInterfuse(Base, ODMRCounterInterface):
    """ This interfuse connects the Arduino controlled VCO and VCA (hardware\arduino_mega2560.py) with
    the National Instruments USB 6210 card (hardware\ni_usb_6210_fast_counter.py).

    Example config for copy-paste:

    scanner_microwave_interfuse:
        module.Class: 'interfuse.scanner_microwave_interfuse.ScannerMicrowaveInterfuse'
        clock_frequency: 100 # in Hz
        # number_of_channels: 1
        baseline_size: 5            # number of count values to calculate a brightness baseline from
        brightness_threshold: 0.8   # if brightness falls below this factor, position will be optimized
        odmr_gate_duration: 200000  # duration of a single odmr signal/reference state in gated mode (in ns)
        connect:
            # fitlogic: 'fitlogic'
            microwave: 'arduino_mega2560'
            confocalscanner1: 'niusb_6210_fastcounter'
            pulsegenerator: 'pulsestreamer'
            optimizerlogic1: 'optimizerlogic'
    """

    # connectors
    microwave = Connector(interface='MicrowaveInterface')
    confocalscanner1 = Connector(interface='ConfocalScannerInterface')
    pulsegenerator = Connector(interface='PulserInterface')
    optimizerlogic1 = Connector(interface='OptimizerLogic')

    # config options
    _clock_frequency = ConfigOption('clock_frequency', 100, missing='warn')
    _baseline_size = ConfigOption('baseline_size', 3, missing='warn')
    _brightness_threshold = ConfigOption('brightness_threshold', 0, missing='warn')
    _odmr_gate_duration = ConfigOption('odmr_gate_duration', 200000, missing='warn')

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        self._odmr_length = None
        self._lock_in_active = False
        self._oversampling = 10
        self._number_of_channels = None
        self._baseline = None

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._microwave_device = self.microwave()
        self._scanning_device = self.confocalscanner1()
        self._pulser_device = self.pulsegenerator()
        self._optimizer_logic = self.optimizerlogic1()

        if not self._scanning_device.is_running:
            try:
                self._scanning_device.start_measure()
            except:
                self.log.error("The counter is not yet running and couldn't be started. Please check what's wrong "
                               "before starting ODMR.")
                return -1

        # Number of ODMR scan channels equals the number of channels of the confocal scanner
        self._number_of_channels = self._scanning_device.number_of_channels

    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        self.log.debug('ODMR counter is shutting down.')

    def set_up_odmr_clock(self, clock_frequency=None, clock_channel=None):
        """ Configures the hardware clock of the NiDAQ card to give the timing.

        @param float clock_frequency: if defined, this sets the frequency of the clock
        @param str clock_channel: if defined, this is the physical channel of the clock

        @return int: error code (0:OK, -1:error)
        """
        if clock_frequency is not None:
            self._clock_frequency = float(clock_frequency)

        return 0

    def set_up_odmr(self, counter_channel=None, photon_source=None,
                    clock_channel=None, odmr_trigger_channel=None):
        """ Configures the actual counter with a given clock.

        @param str counter_channel: if defined, this is the physical channel of the counter
        @param str photon_source: if defined, this is the physical channel where the photons are to count from
        @param str clock_channel: if defined, this specifies the clock for the counter
        @param str odmr_trigger_channel: if defined, this specifies the trigger output for the microwave

        @return int: error code (0:OK, -1:error)
        """
        self.log.info('ScannerMicrowaveInterfuse>set_up_odmr')

        if self.module_state() == 'locked':
            self.log.error('Another odmr is already running, close this one first.')
            return -1

        if self._scanning_device._gated:
            self.pulse_sequence_odmr_on()
        else:
            self.pulse_sequence_odmr_standby()

        # Set microwave power/attenuation before the position optimization
        self._microwave_device.set_mw_pwr(self._microwave_device.mw_sweep_power)

        current_position = self._scanning_device.get_scanner_position()
        if (current_position[0] == 0) and (current_position[1] == 0):
            # Check if confocal scan has been used, at all. If not, a position optimization wouldn't make any sense.
            self.log.error("It looks like you haven't moved to an object in the confocal scan yet.")
            return -1
        else:
            # If it has been used already, start position optimization.
            self.log.info('Position will be optimized before the measurement starts and the counts/s baseline is '
                          'being determined.')
            self.module_state.lock()
            self.optimize_position()
            self.module_state.unlock()

        time.sleep(0.2)  # Might be needed to mitigate too high counts/s at the beginning of the measurement.

        return 0

    def set_odmr_length(self, length=100):
        """ Sets up the trigger sequence for the ODMR and the triggered microwave.

        @param int length: length of microwave sweep in pixel

        @return int: error code (0:OK, -1:error)
        """
        self._odmr_length = length
        return 0

    def count_odmr(self, length=100):
        """ Sweeps the microwave and returns the counts on that sweep.

        @param int length: length of microwave sweep in pixel

        @return float[]: the photon counts per second
        """
        if self.module_state() == 'locked':
            self.log.error('A scan_line is already running, close this one first.')
            return -1

        # Warning that photon counter must be started first, otherwise ODMR scan will crash completely
        if not self._scanning_device.is_running:
            try:
                self._scanning_device.start_measure()
            except:
                self.log.error("The counter is not yet running and couldn't be started. Please check what's wrong "
                               "before starting ODMR.")
                return -1

        self.module_state.lock()

        # Number of frequencies that will be swept through
        self._odmr_length = length

        # Prepare a list of frequencies that will be swept through
        frq_list = list(np.arange(self._microwave_device.mw_start_freq,
                                  self._microwave_device.mw_stop_freq + self._microwave_device.mw_step_freq,
                                  self._microwave_device.mw_step_freq))

        # Shuffle frequency list for each sweep to mitigate systematic measurement errors
        random.shuffle(frq_list)

        # Prepare array for photon counts
        count_data = np.zeros((length, self._number_of_channels))

        # Set microwave power/attenuation
        self._microwave_device.set_mw_pwr(self._microwave_device.mw_sweep_power)

        # Measure baseline for the counts/s (of the Sum channel), if there is not yet one
        if self._baseline is None:
            # Variable for the baseline counts
            baseline_counts = np.zeros(self._baseline_size)

            # Measure the count/s as often as set in 'baseline_size' in config file
            for count_step in range(self._baseline_size):
                baseline_counts[count_step] = self._scanning_device.read_count_diff(1 / self._clock_frequency)[2]

            # The median will be the baseline for the counts/s
            self._baseline = np.median(baseline_counts)

            # Share the value with the counter hardware module as well,
            # to make it available for other modules using the counter.
            self._scanning_device._opt_threshold = self._brightness_threshold * self._baseline

        # Do the actual ODMR scan
        for frq_step in range(length):
            # set the new mw frequency
            self._microwave_device.set_mw_frq(frq_list[frq_step])

            # Count the photons
            count_data[frq_step] = self._scanning_device.read_count_diff(1 / self._clock_frequency)

            # If counts become too small, the position has to be optimized again
            if self.optimization_needed(self._baseline, count_data[frq_step]):
                self.log.info('Position will be optimized as counts/s dropped below threshold.')
                self.optimize_position()

            # ToDo: Could also be programmed more dynamically to suit any number of channels
            # In gated mode: last channel is not the sum but the ratio of signal and reference
            if self._scanning_device._gated:
                count_data[frq_step][2] = count_data[frq_step][0] / count_data[frq_step][1]

        # Sort the count data again by ascending frequency:
        # Make 2D array from frequency list
        frq_list = np.array(frq_list)[np.newaxis]
        # Transpose the list
        frq_list = frq_list.T
        # Use frequency as first col in a count_data matrix with the count_data
        frq_list = np.concatenate((frq_list, count_data), axis=1)
        # Sort list by frequency
        frq_list = frq_list[np.argsort(frq_list[:, 0])]
        # Throw away frequencies again
        count_data = frq_list[:, 1:]

        # Transpose count list as ODMR scan needs it that way
        ret = count_data.T

        self.module_state.unlock()
        return False, ret

    def close_odmr(self):
        """ Closes the odmr and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        self.log.info('ScannerMicrowaveInterfuse>close_odmr')
        self._baseline = None

        return 0

    def close_odmr_clock(self):
        """ Closes the odmr and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        # Return to initial PulseGenerator state (Laser, MW and Gates on)
        if self._scanning_device._gated:
            self.pulse_sequence_odmr_standby()

        self.log.info('ScannerMicrowaveInterfuse>close_odmr_clock')

        return 0

    def get_odmr_channels(self):
        """ Return a list of channel names.

        @return list(str): channels recorded during ODMR measurement
        """
        if self._scanning_device._gated:
            # In Gated counting we have Signal, Reference and Ratio of both
            return ['Signal', 'Reference', 'Sig/Ref', 'ODMR-Threshold']
        else:
            # Number of ODMR scan channels equals the number of channels of the confocal scanner
            return self._scanning_device.scanner_count_channels

    @property
    def oversampling(self):
        return self._oversampling

    @oversampling.setter
    def oversampling(self, val):
        if not isinstance(val, (int, float)):
            self.log.error('oversampling has to be int of float.')
        else:
            self._oversampling = int(val)

    @property
    def lock_in_active(self):
        return self._lock_in_active

    @lock_in_active.setter
    def lock_in_active(self, val):
        if not isinstance(val, bool):
            self.log.error('lock_in_active has to be boolean.')
        else:
            self._lock_in_active = val
            if self._lock_in_active:
                self.log.warn('Lock-In is not implemented')

    # =================== Own Commands ========================
    def optimization_needed(self, baseline, current_counts):
        """ Check if position has to be optimized as the counts/s have become too small.

         @return boolean: True/False
         """
        optimization_needed = False

        # ToDo: Could also be programmed more dynamically to suit any number of channels
        # Check if counts/s have dropped below threshold
        if (current_counts[0] + current_counts[1]) < (self._brightness_threshold * baseline):
            optimization_needed = True

        return optimization_needed

    def optimize_position(self):
        if self._scanning_device._gated:
            # Return to initial PulseGenerator state (Laser, MW and Gates on)
            self.pulse_sequence_odmr_standby()

        # Set state to unlocked, otherwise Qudi will not perform optimization
        self.module_state.unlock()
        # start optimization, 'confocalgui' is needed to pass the new position to the confocalgui and scanner again
        self._optimizer_logic.start_refocus(self._scanning_device.get_scanner_position(), 'confocalgui')
        # wait up to 20 s until optimization is done
        for lock_test in range(21):
            if self._optimizer_logic.module_state() == 'locked':
                time.sleep(1)
                if lock_test == 20:
                    self.log.error('The optimization needs more than 20 seconds. This is not healthy!')
        self.module_state.lock()

        if self._scanning_device._gated:
            # Return to Pulse Sequence for gated counting again
            self.pulse_sequence_odmr_on()

    def pulse_sequence_odmr_on(self):
        """
        Activate the Pulse Streamer sequence for gated odmr scan:
        Laser on, MW, Gate 1 and 2 oscillating between on/off
        """
        ch_laser = self._pulser_device._laser_channel
        ch_micro = self._pulser_device._uw_x_channel
        ch_gt_sg = self._pulser_device._gate_counter_1
        ch_gt_rf = self._pulser_device._gate_counter_2

        duration = self._odmr_gate_duration     # Duration of a single pulse state in ns

        # Creating the PulseStreamer pattern for gated odmr scan
        pulse_pattern_laser = [(duration, 1), (duration, 1)]
        pulse_pattern_micro = [(duration, 0), (duration, 1)]
        pulse_pattern_gt_sg = [(duration, 0), (duration, 1)]
        pulse_pattern_gt_rf = [(duration, 1), (duration, 0)]

        # Assigning the PulseStreamer pattern for gated odmr scan to the PulseStreamer channels
        pulse_seq_odmr = self._pulser_device.pulse_streamer.createSequence()
        pulse_seq_odmr.setDigital(ch_laser, pulse_pattern_laser)
        pulse_seq_odmr.setDigital(ch_micro, pulse_pattern_micro)
        pulse_seq_odmr.setDigital(ch_gt_sg, pulse_pattern_gt_sg)
        pulse_seq_odmr.setDigital(ch_gt_rf, pulse_pattern_gt_rf)

        # Send the Sequence to the Pulse Generator and repeat it infinitely
        n_runs = self._pulser_device.pulse_streamer.REPEAT_INFINITELY
        self._pulser_device.pulse_streamer.stream(pulse_seq_odmr, n_runs)

        # Pulse Generator might need shortly until sequence is up and running
        time.sleep(2)

    def pulse_sequence_odmr_standby(self):
        """
        Activate the Pulse Streamer sequence for optimization and after the ODMR:
        Laser, Gate 1 and 2 on; MW oscillating between on/off
        """
        # read channels from configuration file
        ch_laser = self._pulser_device._laser_channel
        ch_micro = self._pulser_device._uw_x_channel
        ch_gt_sg = self._pulser_device._gate_counter_1
        ch_gt_rf = self._pulser_device._gate_counter_2

        # Duration of a single pulse state in ns
        duration = self._odmr_gate_duration

        # Creating the PulseStreamer pattern for gated odmr scan
        pulse_pattern_laser = [(duration, 1), (duration, 1)]
        pulse_pattern_micro = [(duration, 0), (duration, 1)]
        pulse_pattern_gt_sg = [(duration, 1), (duration, 1)]
        pulse_pattern_gt_rf = [(duration, 1), (duration, 1)]

        # Assigning the PulseStreamer pattern for gated odmr scan to the PulseStreamer channels
        pulse_seq_odmr = self._pulser_device.pulse_streamer.createSequence()
        pulse_seq_odmr.setDigital(ch_laser, pulse_pattern_laser)
        pulse_seq_odmr.setDigital(ch_micro, pulse_pattern_micro)
        pulse_seq_odmr.setDigital(ch_gt_sg, pulse_pattern_gt_sg)
        pulse_seq_odmr.setDigital(ch_gt_rf, pulse_pattern_gt_rf)

        # Send the Sequence to the Pulse Generator and repeat it infinitely
        n_runs = self._pulser_device.pulse_streamer.REPEAT_INFINITELY
        self._pulser_device.pulse_streamer.stream(pulse_seq_odmr, n_runs)

        # Pulse Generator might need shortly until sequence is up and running
        time.sleep(1)
