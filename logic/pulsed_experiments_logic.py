# -*- coding: utf-8 -*-

"""
This file contains the Qudi Logic to control Pulsed Experiments.
It was derived from nuclear operations logic (originally by Alexander Stark) and
adapted to its new purpose by Oliver Gerull.

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

import datetime
import numpy as np
import time
import random

from collections import OrderedDict
from core.connector import Connector
from core.configoption import ConfigOption
from core.statusvariable import StatusVar
from core.util.mutex import Mutex
from logic.generic_logic import GenericLogic
from qtpy import QtCore


class PulsedExperimentsLogic(GenericLogic):
    """ A higher order logic, which combines several lower class logic modules
        in order to perform measurements and manipulations of nuclear spins.

    """

    # declare connectors
    microwave = Connector(interface='MicrowaveInterface')
    confocalscanner = Connector(interface='ConfocalScannerInterface')
    pulsegenerator = Connector(interface='PulserInterface')
    optimizerlogic = Connector(interface='OptimizerLogic')
    savelogic = Connector(interface='SaveLogic')
    fitlogic = Connector(interface='FitLogic')

    # config options
    _baseline_size = ConfigOption('baseline_size', 3, missing='warn')
    _brightness_threshold = ConfigOption('brightness_threshold', 0.8, missing='warn')

    # status vars
    electron_rabi_periode = StatusVar('electron_rabi_periode', 1800e-9)  # in s

    # pulser microwave:
    pulser_mw_freq = StatusVar('pulser_mw_freq', 200e6)  # in Hz
    pulser_mw_amp = StatusVar('pulser_mw_amp', 2.25)  # in V
    pulser_mw_ch = StatusVar('pulser_mw_ch', 1)
    pulser_mw_length = StatusVar('pulser_mw_length', 200e-6)  # in s

    # pulser APD gate channels:
    apd1_gt_ch = StatusVar('apd1_gt_ch', 6)
    apd2_gt_ch = StatusVar('apd2_gt_ch', 7)

    # pulser rf:
    pulser_rf_freq0 = StatusVar('pulser_rf_freq0', 6.32e6)  # in Hz
    pulser_rf_amp0 = StatusVar('pulser_rf_amp0', 0.1)
    pulsed_rabi_period1 = StatusVar('pulsed_rabi_period1', 30e-6)  # in s

    # laser options:
    pulser_laser_ch = StatusVar('pulser_laser_ch', 0)

    # delays:
    laser_delay = StatusVar('laser_delay', 0)  # in ns
    micro_delay = StatusVar('micro_delay', 870)  # in ns
    gate_sg_delay = StatusVar('gate_sg_delay', 860)  # in ns
    gate_rf_delay = StatusVar('gate_rf_delay', 860)  # in ns

    # measurement parameters:
    current_meas_asset_name = StatusVar('current_meas_asset_name', 'Pulsed_Rabi')
    x_axis_start = StatusVar('x_axis_start', 0e-9)  # in s
    x_axis_stop = StatusVar('x_axis_stop', 500e-9)  # in s
    x_axis_stepwidth = StatusVar('x_axis_stepwidth', 10e-9)  # in s
    integration_time = StatusVar('integration_time', 0.2)  # in s
    dD_dT = StatusVar('dD/dT', 65.4e3)  # in Hz/K
    T_0 = StatusVar('T_0', 48063)  # in K

    # How often the measurement should be repeated.
    num_of_meas_runs = StatusVar('num_of_meas_runs', 0)

    # Microwave measurement parameters:
    mw_cw_freq = StatusVar('mw_cw_freq', 2.876e9)  # in Hz
    mw_cw_power = StatusVar('mw_cw_power', -7)  # in dBm

    # temperature measurement: frequencies
    t_meas_frq_1 = StatusVar('t_meas_frq_1', 2.843e9)  # in Hz
    t_meas_frq_2 = StatusVar('t_meas_frq_2', 2.847e9)  # in Hz
    t_meas_frq_3 = StatusVar('t_meas_frq_3', 2.853e9)  # in Hz
    t_meas_frq_4 = StatusVar('t_meas_frq_4', 2.857e9)  # in Hz
    t_meas_frq_5 = StatusVar('t_meas_frq_5', 0.0)  # in Hz
    t_meas_frq_6 = StatusVar('t_meas_frq_6', 0.0)  # in Hz

    # temperature measurement: How many measurements should be averaged?
    num_of_meas_average = StatusVar('num_of_meas_average', 1)

    # temperature measurement: Plot Temperature or Normalized intensity?
    tempscan_plot_type = StatusVar('tempscan_plot_type', 'Temperature')

    # signals
    sigNextMeasPoint = QtCore.Signal()
    sigCurrMeasPointUpdated = QtCore.Signal()
    sigMeasurementStopped = QtCore.Signal()

    sigMeasStarted = QtCore.Signal()

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        self.log.debug('The following configuration was found.')

        # checking for the right configuration
        for key in config.keys():
            self.log.info('{0}: {1}'.format(key, config[key]))

        self._number_of_channels = None
        self._baseline = None

        self.threadlock = Mutex()

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        # establish the access to all connectors:
        self._save_logic = self.savelogic()

        self._microwave_device = self.microwave()
        self._scanning_device = self.confocalscanner()
        self._pulser_device = self.pulsegenerator()
        self._optimizer_logic = self.optimizerlogic()
        self._fit_logic = self.fitlogic()

        # current measurement information:
        self.current_meas_point = self.x_axis_start
        self.current_meas_index = 0
        self.num_of_current_meas_runs = 0
        self.elapsed_time = 0
        self.start_time = datetime.datetime.now()
        self.temp_meas_type = None
        self.delta_omega = None

        self.eternal_mode = False
        self._stop_requested = False

        # Perform initialization routines:
        self.initialize_x_axis()
        self.initialize_y_axis()

        # Fit result (for the GUI)
        self.fit_result = None

        # connect signals:
        self.sigNextMeasPoint.connect(self._meas_point_loop, QtCore.Qt.QueuedConnection)

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
        """ Deactivate the module properly.
        """
        return

    def initialize_x_axis(self):
        """ Initialize the x axis. """

        if self.current_meas_asset_name in ['Pulsed_Rabi']:
            self.x_axis_list = np.arange(self.x_axis_start, self.x_axis_stop, self.x_axis_stepwidth)

            # Make sure that stop value is included, if stepwidth includes it
            # (because np.arange does not always work consistently)
            if int(self.x_axis_list[-1]) == int(self.x_axis_stop - self.x_axis_stepwidth):
                self.x_axis_list = np.append(self.x_axis_list, self.x_axis_stop)
            self.x_axis_end = self.x_axis_list[-1]

        elif self.current_meas_asset_name in ['Temperature_Scan']:
            self.x_axis_list = np.empty(shape=0, dtype=int)

            # the x axis (for the intensities) is given by the chosen frequencies from the GUI
            # ignore frequencies that are still 0 (i.e. not configured in the GUI)
            if self.t_meas_frq_1 > 0:
                self.x_axis_list = np.append(self.x_axis_list, self.t_meas_frq_1)
            if self.t_meas_frq_2 > 0:
                self.x_axis_list = np.append(self.x_axis_list, self.t_meas_frq_2)
            if self.t_meas_frq_3 > 0:
                self.x_axis_list = np.append(self.x_axis_list, self.t_meas_frq_3)
            if self.t_meas_frq_4 > 0:
                self.x_axis_list = np.append(self.x_axis_list, self.t_meas_frq_4)
            if self.t_meas_frq_5 > 0:
                self.x_axis_list = np.append(self.x_axis_list, self.t_meas_frq_5)
            if self.t_meas_frq_6 > 0:
                self.x_axis_list = np.append(self.x_axis_list, self.t_meas_frq_6)

            self.x_axis_list.sort()

            self.time_axis_list = np.empty(shape=0)
            self.temp_axis_list = np.empty(shape=0)

        # Set the (randomized) order of the measurements to mitigate systematic measurement errors
        self.current_meas_order = np.arange(len(self.x_axis_list))
        random.shuffle(self.current_meas_order)

        self.current_meas_point = self.x_axis_list[self.current_meas_order[0]]

    def initialize_y_axis(self):
        """ Initialize the y axis. """

        self.y_axis_list = np.ones(self.x_axis_list.shape)  # y axis where current data are stored
        self.y_axis_list_sig = np.ones(self.x_axis_list.shape)  # y axis where current data are stored
        self.y_axis_list_ref = np.ones(self.x_axis_list.shape)  # y axis where current data are stored

        # here all consecutive measurements are saved, where the
        # self.num_of_meas_runs determines the measurement index for the row.
        self.y_axis_matrix = np.ones((1, len(self.x_axis_list)))
        self.y_axis_matrix_sig = np.ones((1, len(self.x_axis_list)))
        self.y_axis_matrix_ref = np.ones((1, len(self.x_axis_list)))

    def start_pulsed_meas(self, continue_meas=False):
        """ Start the pulsed experiment measurement. """

        self._stop_requested = False
        self.temp_meas_type = None

        if not continue_meas:
            # Preparation of measurement:
            # Set microwave power/attenuation
            self._microwave_device.set_mw_pwr(self.mw_cw_power)
            # Set microwave frequency
            self._microwave_device.set_mw_frq(self.mw_cw_freq)

            self.initialize_x_axis()
            self.initialize_y_axis()
            self.fit_result = None

            self.current_meas_index = 0
            self.sigCurrMeasPointUpdated.emit()
            self.num_of_current_meas_runs = 0

            # Prepare array for photon counts
            self.count_data = np.zeros((len(self.x_axis_list), self._number_of_channels))

            # if number of measurements == 0 then don't stop the measurement
            if self.num_of_meas_runs == 0:
                self.eternal_mode = True
            else:
                self.eternal_mode = False

            self.elapsed_time = 0.
            self.start_time = datetime.datetime.now()

            # Load the measurement protocol with the new current_meas_point
            self.load_pulse_seq(self.current_meas_asset_name, pulsed=True)

            # Optimize position before the measurement
            current_position = self._scanning_device.get_scanner_position()
            if (current_position[0] == 0) and (current_position[1] == 0):
                self.log.error("It looks like you haven't moved to an object in the confocal scan yet.")
                return -1

            if self.current_meas_asset_name in ['Temperature_Scan']:
                if len(self.x_axis_list) != 6:
                    self.log.error(
                        'Please make sure to provide 6 frequencies. Temperature measurement is not yet '
                        'implemented for less frequencies. Thus, no temperature will be calculated.')
                else:
                    self.delta_omega = self.x_axis_list[5] - self.x_axis_list[4]

                    if self.delta_omega != self.x_axis_list[4] - self.x_axis_list[3] or \
                       self.delta_omega != self.x_axis_list[2] - self.x_axis_list[1] or \
                       self.delta_omega != self.x_axis_list[1] - self.x_axis_list[0]:
                        self.log.error(
                            'When providing 6 frequencies, frequencies 1-3 and 4-6 must have the same distance '
                            'from each other. Thus, no temperature will be calculated.')
                    else:
                        self.temp_meas_type = 'six_point_method'

        else:
            # Optimize position before the measurement continues
            self.log.info('Position will be optimized before the measurement starts/continues.')
            self.module_state.lock()
            self.optimize_position()
            self.module_state.unlock()

        time.sleep(0.2)  # Might be needed to mitigate too high counts/s at the beginning of the measurement.

        self.module_state.lock()
        self.sigMeasStarted.emit()
        self.sigNextMeasPoint.emit()

    def _meas_point_loop(self):
        """ Run this loop continuously until an abort criterium is reached. """

        if self._stop_requested:
            with self.threadlock:
                # end measurement and switch all devices off
                self.stopRequested = False
                self.module_state.unlock()

                # Load Pulse Sequence with Laser and Gates continuously on
                self.load_pulse_seq(self.current_meas_asset_name, pulsed=False)

                # emit all needed signals for the update:
                self.sigCurrMeasPointUpdated.emit()
                self.sigMeasurementStopped.emit()
                return
        else:
            self.elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()

            # this routine will return a desired measurement value and the
            # measurement parameters, which belong to it.
            curr_meas_points = self._get_meas_point(self.current_meas_asset_name)

            if self.current_meas_index == 0 and self.num_of_current_meas_runs == 0:
                # Baseline will be determined just at this point.
                # Reason: In this module (pulsed experiments) the baseline is too small at the beginning.
                # Reason: We don't know, yet.
                self.determine_baseline()

            # If counts become too small, the position has to be optimized again
            if self.optimization_needed(self._baseline, curr_meas_points):
                self.log.info('Position will be optimized as counts/s dropped below threshold.')
                self.optimize_position()

            # this routine will handle the saving and storing of the measurement results:
            self._set_meas_point(num_of_meas_runs=self.num_of_current_meas_runs,
                                 meas_index=self.current_meas_index,
                                 meas_points=curr_meas_points)

            if self._stop_requested:
                self.sigNextMeasPoint.emit()
                return

            # increment the measurement index or set it back to zero if it exceed
            # the maximal number of x axis measurement points. The measurement index
            # will be used for the next measurement
            if self.current_meas_index + 1 >= len(self.x_axis_list):
                self.current_meas_index = 0

                # If the next measurement run begins, add a new matrix line to the
                # self.y_axis_matrix
                self.num_of_current_meas_runs += 1

                new_row = np.zeros(len(self.x_axis_list))

                # that vertical stack command behaves similar to the append method
                # in python lists, where the new_row will be appended to the matrix:
                self.y_axis_matrix = np.vstack((self.y_axis_matrix, new_row))
                self.y_axis_matrix_sig = np.vstack((self.y_axis_matrix_sig, new_row))
                self.y_axis_matrix_ref = np.vstack((self.y_axis_matrix_ref, new_row))

                # Randomize the measurement order again to mitigate systematic measurement errors
                random.shuffle(self.current_meas_order)

            else:
                self.current_meas_index += 1

            # check if measurement is at the end, and if not, adjust the measurement
            # sequence to the next measurement point.
            if self.eternal_mode or (self.num_of_current_meas_runs < self.num_of_meas_runs):

                # take the next measurement index from the x axis as the current
                # measurement point:
                self.current_meas_point = self.x_axis_list[self.current_meas_order[self.current_meas_index]]

                # Load the measurement protocol with the new current_meas_point
                self.load_pulse_seq(self.current_meas_asset_name, pulsed=True)
            else:
                # Load Pulse Sequence with Laser and Gates continuously on
                self.load_pulse_seq(self.current_meas_asset_name, pulsed=False)

            self.sigNextMeasPoint.emit()

    def _set_meas_point(self, num_of_meas_runs, meas_index, meas_points):
        """ Handle the proper setting of the current meas_point and store all
            the additional measurement parameter.

        @param int meas_index:
        @param int num_of_meas_runs
        @param float meas_points:
        @return:
        """
        # one matrix contains all the measured values, the other one contains
        # all the parameters for the specified measurement point:
        self.y_axis_matrix[num_of_meas_runs, self.current_meas_order[meas_index]] = meas_points[2]
        self.y_axis_matrix_sig[num_of_meas_runs, self.current_meas_order[meas_index]] = meas_points[0]
        self.y_axis_matrix_ref[num_of_meas_runs, self.current_meas_order[meas_index]] = meas_points[1]

        if self.current_meas_asset_name in ['Temperature_Scan']:

            if self.num_of_current_meas_runs < self.num_of_meas_average:
                # at the beginning the y_axis_list contains latest measurement values
                self.y_axis_list[self.current_meas_order[meas_index]] = \
                    self.y_axis_matrix[num_of_meas_runs, self.current_meas_order[meas_index]]
                self.y_axis_list_sig[self.current_meas_order[meas_index]] = \
                    self.y_axis_matrix_sig[num_of_meas_runs, self.current_meas_order[meas_index]]
                self.y_axis_list_ref[self.current_meas_order[meas_index]] = \
                    self.y_axis_matrix_ref[num_of_meas_runs, self.current_meas_order[meas_index]]
            else:
                # Wait until enough measurements have been made to calculate a mean value.

                # If counts for all frequencies have been measured, calculate temperature
                if self.current_meas_index + 1 >= len(self.x_axis_list):
                    # the y_axis_list contains the average over the last measurement values
                    # (average might be needed when fit is used and leads to too many wrong values due to noise)
                    self.y_axis_list = self.y_axis_matrix[-self.num_of_meas_average:, :].mean(axis=0)

                    temperature = 0

                    if self.temp_meas_type == 'six_point_method':
                        # Calculate temperature as described in PhysRevResearch.2.043415
                        intensity = self.y_axis_list

                        delta_omega_4pt_1 = self.delta_omega * (
                                    (intensity[0] + intensity[2]) - (intensity[3] + intensity[5])) / (
                                    (intensity[0] - intensity[2]) - (intensity[3] - intensity[5]))
                        delta_omega_4pt_2 = (self.delta_omega / 2) * (
                                    (intensity[0] + intensity[1]) - (intensity[4] + intensity[5])) / (
                                    (intensity[0] - intensity[1]) - (intensity[4] - intensity[5]))
                        delta_omega_4pt_3 = (self.delta_omega / 2) * (
                                    (intensity[1] + intensity[2]) - (intensity[3] + intensity[4])) / (
                                    (intensity[1] - intensity[2]) - (intensity[3] - intensity[4]))
                        delta_omega_6pt = (delta_omega_4pt_1 + delta_omega_4pt_2 + delta_omega_4pt_3) / 3

                        delta_temperature_6pt = - delta_omega_6pt / self.dD_dT

                        temperature = self.T_0 + delta_temperature_6pt

                    elif self.temp_meas_type == 'lorentzian_fit':
                        # Make Lorentzian fit
                        self.fit_result = self._fit_logic.make_lorentzian_fit(
                            x_axis=self.x_axis_list,
                            data=self.y_axis_list,
                            estimator=self._fit_logic.estimate_lorentzian_dip)

                        # Calculate the temperature from the lorentzian dip
                        fit_center = self.fit_result.params['center'].value

                        # Check if fit_center is in inner 50 % range of the given frequencies, otherwise fit went wrong
                        quarter_width = (max(self.x_axis_list) - min(self.x_axis_list)) / 4
                        if min(self.x_axis_list) + quarter_width < fit_center < max(self.x_axis_list) - quarter_width:
                            temperature = self.T_0 - ((1 / self.dD_dT) * fit_center) - 273.15

                    elif self.temp_meas_type == 'double_lorentzian_fit':
                        # Make Double Lorentzian fit
                        self.fit_result = self._fit_logic.make_lorentziandouble_fit(
                            x_axis=self.x_axis_list,
                            data=self.y_axis_list,
                            estimator=self._fit_logic.estimate_lorentziandouble_dip)

                        # Calculate the temperature from the lorentzian dip
                        # ToDo: center might be the wrong parameter
                        fit_center = self.fit_result.params['center'].value

                        # Check if fit_center is in inner 50 % range of the given frequencies, otherwise fit went wrong
                        quarter_width = (max(self.x_axis_list) - min(self.x_axis_list)) / 4
                        if min(self.x_axis_list) + quarter_width < fit_center < max(self.x_axis_list) - quarter_width:
                            temperature = self.T_0 - ((1 / self.dD_dT) * fit_center) - 273.15

                    # Save time and temperature
                    self.time_axis_list = np.append(self.time_axis_list, self.elapsed_time)
                    self.temp_axis_list = np.append(self.temp_axis_list, temperature)

        else:
            # the y_axis_list contains the summed and averaged values for each measurement index:
            self.y_axis_list[self.current_meas_order[meas_index]] = \
                self.y_axis_matrix[:, self.current_meas_order[meas_index]].mean()
            self.y_axis_list_sig[self.current_meas_order[meas_index]] = \
                self.y_axis_matrix_sig[:, self.current_meas_order[meas_index]].mean()
            self.y_axis_list_ref[self.current_meas_order[meas_index]] = \
                self.y_axis_matrix_ref[:, self.current_meas_order[meas_index]].mean()

        self.sigCurrMeasPointUpdated.emit()

    def _get_meas_point(self, meas_type):
        """ Start the actual measurement

        And perform the measurement with that routine.
        @return tuple (float, dict):
        """
        # Counter measurement
        count_data = self._scanning_device.read_count_diff_once(self.integration_time)
        count_data[2] = count_data[0] / count_data[1]

        return count_data

    def stop_pulsed_meas(self):
        """ Stop the Pulsed Experiment Measurement.

        @return int: error code (0:OK, -1:error)
        """
        with self.threadlock:
            self._stop_requested = True

        return 0

    def get_meas_type_list(self):
        return ['Pulsed_Rabi', 'Temperature_Scan']

    def get_tempscan_plot_list(self):
        """
        Return possible plot types according to the given measurement type
        """
        if self.current_meas_asset_name in ['Temperature_Scan']:
            return ['Normalized intensity', 'Temperature']
        else:
            return['Normalized intensity']

    def load_pulse_seq(self, meas_type, pulsed=True):
        """
        Prepare all measurement protocols for the specified measurement type
        and load it to the Pulse Streamer

        @param str meas_type: a measurement type from the list get_meas_type_list
        """
        # Read channels from the configuration file
        ch_laser = self.pulser_laser_ch
        ch_micro = self.pulser_mw_ch
        ch_gt_sg = self.apd1_gt_ch
        ch_gt_rf = self.apd2_gt_ch

        # Read delays (between pulse signal and reaction of the device) from the configuration file
        d_laser = int(self.laser_delay)
        d_micro = int(self.micro_delay)
        d_gt_sg = int(self.gate_sg_delay)
        d_gt_rf = int(self.gate_rf_delay)
        d_max = max(d_laser, d_micro, d_gt_sg, d_gt_rf)

        if meas_type == 'Pulsed_Rabi':
            # Durations of each pulser state in ns
            tau = self.current_meas_point * 1e9
            tau_max = self.x_axis_end * 1e9
            durations = (0, 3000, 300, 50, 500, int(tau), 30, 300, 30, int(tau_max - tau), 0)

            # Creating the PulseStreamer pattern
            if pulsed:
                pulse_states_laser = (1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1)
                pulse_states_micro = (0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)
                pulse_states_gt_sg = (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
                pulse_states_gt_rf = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
            else:
                pulse_states_laser = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
                pulse_states_micro = (1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)
                pulse_states_gt_sg = (1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)
                pulse_states_gt_rf = (1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)

        elif meas_type == 'Temperature_Scan':
            # Set MW frequency
            mw_freq = self.current_meas_point
            self._microwave_device.set_mw_frq(mw_freq)

            # Durations of each pulser state in ns
            durations = (0, self.pulser_mw_length*1e9, self.pulser_mw_length*1e9, 0)

            # Creating the PulseStreamer pattern
            if pulsed:
                pulse_states_laser = (1, 1, 1, 1)
                pulse_states_micro = (0, 1, 0, 0)
                pulse_states_gt_sg = (0, 1, 0, 0)
                pulse_states_gt_rf = (0, 0, 1, 0)
            else:
                pulse_states_laser = (1, 1, 1, 1)
                pulse_states_micro = (0, 1, 0, 0)
                pulse_states_gt_sg = (1, 1, 1, 1)
                pulse_states_gt_rf = (1, 1, 1, 1)

        # Extend Patterns according to the delays:
        # To manipulate single entries, the durations have to be lists not tuples
        durations_laser = list(durations)
        durations_micro = list(durations)
        durations_gt_sg = list(durations)
        durations_gt_rf = list(durations)

        # The first state is supposed to move each pattern forward (earlier) by the respective delay
        durations_laser[0] = d_laser
        durations_micro[0] = d_micro
        durations_gt_sg[0] = d_gt_sg
        durations_gt_rf[0] = d_gt_rf

        # The last state is supposed to extend each pattern by the respective delay (to make them same length again)
        durations_laser[-1] = d_max - d_laser
        durations_micro[-1] = d_max - d_micro
        durations_gt_sg[-1] = d_max - d_gt_sg
        durations_gt_rf[-1] = d_max - d_gt_rf

        # Turn durations into tuples again
        durations_laser = tuple(durations_laser)
        durations_micro = tuple(durations_micro)
        durations_gt_sg = tuple(durations_gt_sg)
        durations_gt_rf = tuple(durations_gt_rf)

        # Bring them into the right format
        pulse_pattern_laser = np.vstack((durations_laser, pulse_states_laser))
        pulse_pattern_laser = list(zip(*pulse_pattern_laser))

        pulse_pattern_micro = np.vstack((durations_micro, pulse_states_micro))
        pulse_pattern_micro = list(zip(*pulse_pattern_micro))

        pulse_pattern_gt_sg = np.vstack((durations_gt_sg, pulse_states_gt_sg))
        pulse_pattern_gt_sg = list(zip(*pulse_pattern_gt_sg))

        pulse_pattern_gt_rf = np.vstack((durations_gt_rf, pulse_states_gt_rf))
        pulse_pattern_gt_rf = list(zip(*pulse_pattern_gt_rf))

        # Assign the PulseStreamer pattern to the PulseStreamer channels
        pulse_seq = self._pulser_device.pulse_streamer.createSequence()
        pulse_seq.setDigital(ch_laser, pulse_pattern_laser)
        pulse_seq.setDigital(ch_micro, pulse_pattern_micro)
        pulse_seq.setDigital(ch_gt_sg, pulse_pattern_gt_sg)
        pulse_seq.setDigital(ch_gt_rf, pulse_pattern_gt_rf)

        # Show the sequence to the user (just at the 2nd measurement point, as it would be shown twice at the 1st point)
        if pulsed and (self.current_meas_index == 1 and self.num_of_current_meas_runs == 0):
            print('Sample PulseStreamer sequence (includes delay correction):')
            print('Laser:     ', pulse_pattern_laser)
            print('Microwave: ', pulse_pattern_micro)
            print('Gate (Sig):', pulse_pattern_gt_sg)
            print('Gate (Ref):', pulse_pattern_gt_rf)

        # Send the Sequence to the Pulse Generator and repeat it infinitely
        n_runs = self._pulser_device.pulse_streamer.REPEAT_INFINITELY
        self._pulser_device.pulse_streamer.stream(pulse_seq, n_runs)

    def save_pulsed_experiments_measurement(self, name_tag=None, timestamp=None):
        """ Save the pulsed experiment data.

        @param str name_tag:
        @param object timestamp: datetime.datetime object, from which everything
                                 can be created.
        """
        filepath = self._save_logic.get_path_for_module(module_name='PulsedExperiments')

        if timestamp is None:
            timestamp = datetime.datetime.now()

        if name_tag is not None and len(name_tag) > 0:
            filelabel1 = name_tag + '_pulsed_exp_xy_data'
            filelabel2 = name_tag + '_pulsed_exp_data_y_matrix'
        else:
            filelabel1 = '_pulsed_exp_data'
            filelabel2 = '_pulsed_exp_data_matrix'

        filelabel3 = '_pulsed_exp_temp'

        param = OrderedDict()
        param2 = OrderedDict()
        param['Pulser Laser channel'] = self.pulser_laser_ch
        param['Pulser MW channel'] = self.pulser_mw_ch
        param['Pulser APD1 Gate channel'] = self.apd1_gt_ch
        param['Pulser APD2 Gate channel'] = self.apd2_gt_ch
        param['Laser Delay'] = self.laser_delay
        param['Microwave Delay'] = self.micro_delay
        param['Gate (Sig.) Delay'] = self.gate_sg_delay
        param['Gate (Ref.) Delay'] = self.gate_rf_delay

        data1 = OrderedDict()
        data2 = OrderedDict()
        data3 = OrderedDict()

        # Measurement Parameter:
        param['Measurement Type'] = self.current_meas_asset_name
        if self.current_meas_asset_name in ['Pulsed_Rabi']:
            param['tau start (ns)'] = int(self.x_axis_start * 1e9)
            param['tau stop (ns)'] = int(self.x_axis_stop * 1e9)
            param['tau stepwidth (ns)'] = int(self.x_axis_stepwidth * 1e9)
            param['Integration time (ms)'] = int(self.integration_time * 1e3)
            param['Last measurement point (ns)'] = int(self.current_meas_point * 1e9)

            data1['tau (s),normalized intensity,Signal (Counts/s),Reference (Counts/s)'] = np.vstack((self.x_axis_list, self.y_axis_list, self.y_axis_list_sig, self.y_axis_list_ref)).T

            param2['tau (s)'] = self.x_axis_list
            data2['normalized intensity matrix'] = self.y_axis_matrix

        elif self.current_meas_asset_name in ['Temperature_Scan']:
            param['Pulser MW length (µs)'] = int(self.pulser_mw_length * 1e6)
            param['x axis start (ns)'] = self.x_axis_start * 1e9
            param['x axis step (ns)'] = self.x_axis_stepwidth * 1e9
            param['x axis stop (ns)'] = self.x_axis_stepwidth * 1e9
            param['Integration time (ms)'] = self.integration_time * 1e3
            param['dD/dT (kHz/K)'] = self.dD_dT / 1e3
            param['T_0 (K)'] = self.T_0
            param['Last measurement point'] = self.current_meas_point

            param['Temperature Measurement Frequency 1 (GHz)'] = self.t_meas_frq_1/1e9
            param['Temperature Measurement Frequency 2 (GHz)'] = self.t_meas_frq_2/1e9
            param['Temperature Measurement Frequency 3 (GHz)'] = self.t_meas_frq_3/1e9
            param['Temperature Measurement Frequency 4 (GHz)'] = self.t_meas_frq_4/1e9
            param['Temperature Measurement Frequency 5 (GHz)'] = self.t_meas_frq_5/1e9
            param['Temperature Measurement Frequency 6 (GHz)'] = self.t_meas_frq_6/1e9

            data1['Frq (GHz),last normalized intensity,Signal (Counts/s),Reference (Counts/s)'] = np.vstack(
                (self.x_axis_list, self.y_axis_list, self.y_axis_list_sig, self.y_axis_list_ref)).T
            data3['Time(s),Temperature(°C)'] = np.vstack(
                (self.time_axis_list, self.temp_axis_list)).T

            param2['Frq (GHz)'] = self.x_axis_list
            data2['normalized intensity matrix'] = self.y_axis_matrix

        param['Number of expected measurement points per run'] = len(self.x_axis_list)
        param['Number of expected measurement runs'] = self.num_of_meas_runs
        param['Number of actual measurement runs'] = self.num_of_current_meas_runs

        param['MW freq (GHz)'] = self.mw_cw_freq / 1e9
        param['MW power (dBm)'] = self.mw_cw_power

        param['Elapsed Time (s)'] = self.elapsed_time
        param['Start of measurement'] = self.start_time.strftime('%Y-%m-%d %H:%M:%S')

        self._save_logic.save_data(data1,
                                   filepath=filepath,
                                   parameters=param,
                                   filelabel=filelabel1,
                                   timestamp=timestamp)

        self._save_logic.save_data(data2,
                                   filepath=filepath,
                                   parameters=param2,
                                   filelabel=filelabel2,
                                   timestamp=timestamp)

        if self.current_meas_asset_name in ['Temperature_Scan']:
            self._save_logic.save_data(data3,
                                       filepath=filepath,
                                       parameters=param,
                                       filelabel=filelabel3,
                                       timestamp=timestamp)

        self.log.info('Pulsed Experiment data saved to:\n{0}'.format(filepath))

    # =============================================================================================
    def switch_all_on(self):
        """ Switch on all PulseStreamer outputs
        """
        try:
            self._pulser_device.pulse_streamer.constant(self._pulser_device._all_on_state)
        except:
            self.log.error("The PulseStreamer outputs could not be switched to permanently on.")

    def switch_all_off(self):
        """ Switch on all PulseStreamer outputs
        """
        # Switch off all PulseStreamer Outputs
        try:
            self._pulser_device.pulse_streamer.constant(self._pulser_device._all_off_state)
        except:
            self.log.error("The PulseStreamer outputs could not be switched to permanently off.")

    def optimization_needed(self, baseline, current_counts):
        """ Check if position has to be optimized as the counts/s have become too small.

         @return boolean: True/False
         """
        optimization_needed = False

        # Check if counts/s have dropped below threshold
        if (current_counts[0] + current_counts[1]) < (self._brightness_threshold * baseline):
            optimization_needed = True

        return optimization_needed

    def optimize_position(self):
        # Load Pulse Sequence with Laser and Gates switched on
        self.load_pulse_seq(self.current_meas_asset_name, pulsed=False)

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

        # Load Pulse Sequence with pulsed Laser and Gates
        self.load_pulse_seq(self.current_meas_asset_name, pulsed=True)

    def determine_baseline(self):
        self.log.info('Position will be optimized.')
        self.optimize_position()

        self.log.info('Counts/s baseline will be determined.')

        # Measure baseline for the counts/s (of the Sum channel)
        # Variable for the baseline counts
        baseline_counts = np.zeros(self._baseline_size)

        # Measure the count/s as often as set in 'baseline_size' in config file
        for count_step in range(self._baseline_size):
            baseline_counts[count_step] = self._scanning_device.read_count_diff(self.integration_time)[2]

        # The median will be the baseline for the counts/s
        self._baseline = np.median(baseline_counts)

        # Share the value with the counter hardware module as well,
        # to make it available for other modules using the counter.
        self._scanning_device._opt_threshold = self._brightness_threshold * self._baseline