# -*- coding: utf-8 -*-
"""
This file contains the Qudi GUI module for pulsed ODMR experiments (to measure Rabi oscillations and temperature).
It was derived from nuclear operations GUI  (originally by Alexander Stark) and
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
import os
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np

from core.connector import Connector
from gui.guibase import GUIBase
from qtpy import QtWidgets, QtCore, uic
from gui.colordefs import QudiPalettePale as palette


class PulsedExperimentsMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_pulsed_experiments_gui.ui')

        # Load it
        super(PulsedExperimentsMainWindow, self).__init__()
        uic.loadUi(ui_file, self)
        self.show()


class PulsedExperimentsGui(GUIBase):
    """ This is the main GUI Class for Pulsed Experiments. """

    # declare connectors
    pulsedexperimentslogic = Connector(interface='PulsedExperimentsLogic')
    savelogic = Connector(interface='SaveLogic')

    def on_activate(self):
        """
        This init connects all the graphic modules, which were created in the
        *.ui file and configures the event handling between the modules.
        """

        self._pe_logic = self.pulsedexperimentslogic()
        self._save_logic = self.savelogic()

        # Create the MainWindow to display the GUI
        self._mw = PulsedExperimentsMainWindow()

        # Add save file tag input box
        self._mw.save_tag_LineEdit = QtWidgets.QLineEdit(self._mw)
        self._mw.save_tag_LineEdit.setMaximumWidth(200)
        self._mw.save_tag_LineEdit.setToolTip('Enter a nametag which will be\n'
                                              'added to the filename.')
        self._mw.save_ToolBar.addWidget(self._mw.save_tag_LineEdit)

        # Set the values from the logic to the GUI:

        # Set the pulser parameter:
        self._mw.electron_rabi_periode_DSpinBox.setValue(self._pe_logic.electron_rabi_periode * 1e9)
        self._mw.pulser_mw_freq_DSpinBox.setValue(self._pe_logic.pulser_mw_freq / 1e6)
        self._mw.pulser_mw_amp_DSpinBox.setValue(self._pe_logic.pulser_mw_amp)
        self._mw.pulser_mw_ch_SpinBox.setValue(self._pe_logic.pulser_mw_ch)
        self._mw.apd1_gt_ch_SpinBox.setValue(self._pe_logic.apd1_gt_ch)
        self._mw.apd2_gt_ch_SpinBox.setValue(self._pe_logic.apd2_gt_ch)
        self._mw.pulser_rf_freq0_DSpinBox.setValue(self._pe_logic.pulser_rf_freq0 / 1e6)
        self._mw.pulser_rf_amp0_DSpinBox.setValue(self._pe_logic.pulser_rf_amp0)
        self._mw.pulser_mw_length_DSpinBox.setValue(self._pe_logic.pulser_mw_length * 1e6)
        self._mw.pulser_laser_ch_SpinBox.setValue(self._pe_logic.pulser_laser_ch)

        # Set the delay parameter:
        self._mw.delay_laser_DSpinBox.setValue(self._pe_logic.laser_delay)
        self._mw.delay_microwave_DSpinBox.setValue(self._pe_logic.micro_delay)
        self._mw.delay_gate_sig_DSpinBox.setValue(self._pe_logic.gate_sg_delay)
        self._mw.delay_gate_ref_DSpinBox.setValue(self._pe_logic.gate_rf_delay)

        # set the measurement parameter:
        self._mw.current_meas_asset_name_ComboBox.clear()
        self._mw.current_meas_asset_name_ComboBox.addItems(self._pe_logic.get_meas_type_list())
        if self._pe_logic.current_meas_asset_name != '':
            index = self._mw.current_meas_asset_name_ComboBox.findText(self._pe_logic.current_meas_asset_name,
                                                                       QtCore.Qt.MatchFixedString)
            if index >= 0:
                self._mw.current_meas_asset_name_ComboBox.setCurrentIndex(index)

        self._mw.tempscan_plot_type_ComboBox.clear()
        self._mw.tempscan_plot_type_ComboBox.addItems(self._pe_logic.get_tempscan_plot_list())
        if self._pe_logic.tempscan_plot_type != '':
            index = self._mw.tempscan_plot_type_ComboBox.findText(self._pe_logic.tempscan_plot_type,
                                                                       QtCore.Qt.MatchFixedString)
            if index >= 0:
                self._mw.tempscan_plot_type_ComboBox.setCurrentIndex(index)

        self._mw.x_axis_start_DSpinBox.setValue(self._pe_logic.x_axis_start * 1e9)
        self._mw.x_axis_stop_DSpinBox.setValue(self._pe_logic.x_axis_stop * 1e9)
        self._mw.x_axis_stepwidth_DSpinBox.setValue(self._pe_logic.x_axis_stepwidth * 1e9)
        self._mw.integration_time_DSpinBox.setValue(self._pe_logic.integration_time * 1e3)
        self._mw.num_of_meas_runs_SpinBox.setValue(self._pe_logic.num_of_meas_runs)
        self._mw.dD_dT_DSpinBox.setValue(self._pe_logic.dD_dT / 1e3)
        self._mw.T_0_SpinBox.setValue(self._pe_logic.T_0)
        self._mw.num_of_meas_average_SpinBox.setValue(self._pe_logic.num_of_meas_average)

        # set the mw parameters for measurement
        self._mw.mw_cw_freq_DSpinBox.setValue(self._pe_logic.mw_cw_freq / 1e9)
        self._mw.mw_cw_power_DSpinBox.setValue(self._pe_logic.mw_cw_power)

        # set the temperature measurement frequencies:
        self._mw.t_meas_frq_1_DSpinBox.setValue(self._pe_logic.t_meas_frq_1 / 1e9)
        self._mw.t_meas_frq_2_DSpinBox.setValue(self._pe_logic.t_meas_frq_2 / 1e9)
        self._mw.t_meas_frq_3_DSpinBox.setValue(self._pe_logic.t_meas_frq_3 / 1e9)
        self._mw.t_meas_frq_4_DSpinBox.setValue(self._pe_logic.t_meas_frq_4 / 1e9)
        self._mw.t_meas_frq_5_DSpinBox.setValue(self._pe_logic.t_meas_frq_5 / 1e9)
        self._mw.t_meas_frq_6_DSpinBox.setValue(self._pe_logic.t_meas_frq_6 / 1e9)

        # Create the graphic display for the measurement:
        pen_style = pg.mkPen(width=0.5, style=QtCore.Qt.DotLine)
        self.pulsed_exp_graph = pg.PlotDataItem(self._pe_logic.x_axis_list,
                                                self._pe_logic.y_axis_list,
                                                pen=pen_style,
                                                symbol='o',
                                                symbolSize=3)


        self._mw.pulsed_exp_GraphicsView.addItem(self.pulsed_exp_graph)

        # Add an empty fit that will be changed if a fit is done
        self.fit_image = pg.PlotDataItem(
            x=(),
            y=(),
            pen=pg.mkPen(palette.c2)
        )

        self._mw.pulsed_exp_GraphicsView.addItem(self.fit_image)

        # Set the proper initial display:
        self.current_meas_asset_name_changed()

        # Connect the signals:
        self._mw.current_meas_asset_name_ComboBox.currentIndexChanged.connect(self.current_meas_asset_name_changed)
        self._mw.tempscan_plot_type_ComboBox.currentIndexChanged.connect(self.tempscan_plot_type_changed)

        # adapt the unit according to the

        # Connect the start and stop signals:
        self._mw.action_run_stop.toggled.connect(self.start_stop_measurement)
        self._mw.action_continue.toggled.connect(self.continue_stop_measurement)
        self._mw.action_save.triggered.connect(self.save_measurement)
        self._pe_logic.sigMeasurementStopped.connect(self._update_display_meas_stopped)
        self._mw.action_all_on.triggered.connect(self.switch_all_on)
        self._mw.action_all_off.triggered.connect(self.switch_all_off)

        # Connect graphic update:
        self._pe_logic.sigCurrMeasPointUpdated.connect(self.update_meas_graph)
        self._pe_logic.sigCurrMeasPointUpdated.connect(self.update_meas_parameter)

    def on_deactivate(self):
        """ Reverse steps of activation

        @return int: error code (0:OK, -1:error)
        """
        self._mw.close()

    def show(self):
        """Make window visible and put it above all other windows. """
        QtWidgets.QMainWindow.show(self._mw)
        self._mw.activateWindow()
        self._mw.raise_()

    def start_stop_measurement(self, is_checked):
        """ Manages what happens if pulsed experiments are started/stopped.

        @param bool is_checked: If true measurement is started, if false
                               measurement stops.
        """
        if is_checked:
            # change the axes appearance according to input values:
            self._pe_logic.stop_pulsed_meas()

            self.update_all_logic_parameter()
            self._pe_logic.start_pulsed_meas()
            self._mw.action_continue.setEnabled(False)
        else:
            self._pe_logic.stop_pulsed_meas()
            self._mw.action_continue.setEnabled(True)
            self._mw.action_run_stop.setChecked(False)

    def continue_stop_measurement(self, is_checked):
        """ Manages what happens if pulsed experiments are continued/stopped.

        @param bool is_checked: If true measurement is continued, if false
                               measurement stops.
        """
        if is_checked:
            self._pe_logic.stop_pulsed_meas()
            self._pe_logic.start_pulsed_meas(continue_meas=True)
            self._mw.action_run_stop.setEnabled(False)

        else:
            self._pe_logic.stop_pulsed_meas()
            self._mw.action_run_stop.setEnabled(True)
            self._mw.action_continue.setChecked(False)

    def _update_display_meas_stopped(self):
        """ Update all the displays of the current measurement state and set
            them to stop. """

        self.start_stop_measurement(is_checked=False)
        self.continue_stop_measurement(is_checked=False)

    def current_meas_asset_name_changed(self):
        """ Adapt the input widget to the current measurement sequence. """

        name = self._mw.current_meas_asset_name_ComboBox.currentText()
        self._pe_logic.current_meas_asset_name = name

        if name == 'Pulsed_Rabi':
            self._mw.pulser_mw_length_DSpinBox.setVisible(False)
            self._mw.label_31.setVisible(False)

            self._mw.pulser_rf_freq0_DSpinBox.setVisible(True)
            self._mw.pulser_rf_freq0_Label.setVisible(True)

            self._mw.x_axis_start_Label.setVisible(True)
            self._mw.x_axis_stop_Label.setVisible(True)
            self._mw.x_axis_stepwidth_Label.setVisible(True)
            self._mw.x_axis_start_DSpinBox.setVisible(True)
            self._mw.x_axis_stop_DSpinBox.setVisible(True)
            self._mw.x_axis_stepwidth_DSpinBox.setVisible(True)

            self._mw.dD_dT_Label.setVisible(False)
            self._mw.dD_dT_DSpinBox.setVisible(False)
            self._mw.T_0_Label.setVisible(False)
            self._mw.T_0_SpinBox.setVisible(False)
            self._mw.num_of_meas_average_SpinBox.setVisible(False)
            self._mw.num_of_meas_average_label.setVisible(False)

            self._mw.pulsed_exp_GraphicsView.setLabel(axis='bottom',
                                                      text='tau',
                                                      units='s')
            self._mw.pulsed_exp_GraphicsView.setLabel(axis='left',
                                                      text='normalized intensity',
                                                      units='')

            self._mw.x_axis_start_Label.setText('\u03C4 start (ns)')
            self._mw.x_axis_stop_Label.setText('\u03C4 stop (ns)')
            self._mw.x_axis_stepwidth_Label.setText('\u03C4 stepwidth (ns)')

            self._mw.current_meas_point_Label.setText('Curr meas point (\u00B5s)')

            self._mw.tempscan_plot_type_ComboBox.clear()
            self._mw.tempscan_plot_type_ComboBox.addItems(self._pe_logic.get_tempscan_plot_list())
            if self._pe_logic.tempscan_plot_type != '':
                index = self._mw.tempscan_plot_type_ComboBox.findText(self._pe_logic.tempscan_plot_type,
                                                                      QtCore.Qt.MatchFixedString)
                if index >= 0:
                    self._mw.tempscan_plot_type_ComboBox.setCurrentIndex(index)

        elif name == 'Temperature_Scan':
            self._mw.pulser_mw_length_DSpinBox.setVisible(True)
            self._mw.label_31.setVisible(True)

            self._mw.pulser_rf_freq0_DSpinBox.setVisible(True)
            self._mw.pulser_rf_freq0_Label.setVisible(True)

            self._mw.x_axis_start_Label.setVisible(False)
            self._mw.x_axis_stop_Label.setVisible(False)
            self._mw.x_axis_stepwidth_Label.setVisible(False)
            self._mw.x_axis_start_DSpinBox.setVisible(False)
            self._mw.x_axis_stop_DSpinBox.setVisible(False)
            self._mw.x_axis_stepwidth_DSpinBox.setVisible(False)

            self._mw.dD_dT_Label.setVisible(True)
            self._mw.dD_dT_DSpinBox.setVisible(True)
            self._mw.T_0_Label.setVisible(True)
            self._mw.T_0_SpinBox.setVisible(True)
            self._mw.num_of_meas_average_SpinBox.setVisible(True)
            self._mw.num_of_meas_average_label.setVisible(True)

            self._mw.current_meas_point_Label.setText('Curr meas point (MHz)')

            self._mw.tempscan_plot_type_ComboBox.clear()
            self._mw.tempscan_plot_type_ComboBox.addItems(self._pe_logic.get_tempscan_plot_list())
            if self._pe_logic.tempscan_plot_type != '':
                index = self._mw.tempscan_plot_type_ComboBox.findText(self._pe_logic.tempscan_plot_type,
                                                                      QtCore.Qt.MatchFixedString)
                if index >= 0:
                    self._mw.tempscan_plot_type_ComboBox.setCurrentIndex(index)

            # Update the plot, too
            self.tempscan_plot_type_changed()

    def tempscan_plot_type_changed(self):
        """ Adapt the input widget to the current measurement sequence. """

        type = self._mw.tempscan_plot_type_ComboBox.currentText()
        name = self._mw.current_meas_asset_name_ComboBox.currentText()

        # tell the pulsed_experiments logic about the new plot type
        self._pe_logic.tempscan_plot_type = type

        # Adapt the plot's axis labels to the respective plot type
        if name == 'Temperature_Scan':
            if type == 'Temperature':
                self._mw.pulsed_exp_GraphicsView.setLabel(axis='bottom',
                                                          text='measurement time',
                                                          units="s")
                self._mw.pulsed_exp_GraphicsView.setLabel(axis='left',
                                                          text='temperature',
                                                          units='°C')
            elif type == 'Normalized intensity':
                self._mw.pulsed_exp_GraphicsView.setLabel(axis='bottom',
                                                          text='microwave frequency',
                                                          units="Hz")
                self._mw.pulsed_exp_GraphicsView.setLabel(axis='left',
                                                          text='normalized intensity',
                                                          units='')
        else:
            self._mw.pulsed_exp_GraphicsView.setLabel(axis='bottom',
                                                      text='tau',
                                                      units='s')
            self._mw.pulsed_exp_GraphicsView.setLabel(axis='left',
                                                      text='normalized intensity',
                                                      units='')

    def update_all_logic_parameter(self):
        """ If the measurement is started, update all parameters in the logic.
        """

        # pulser parameter:
        self._pe_logic.electron_rabi_periode = self._mw.electron_rabi_periode_DSpinBox.value() / 1e9
        self._pe_logic.pulser_mw_freq = self._mw.pulser_mw_freq_DSpinBox.value() * 1e6
        self._pe_logic.pulser_mw_amp = self._mw.pulser_mw_amp_DSpinBox.value()
        self._pe_logic.pulser_mw_ch = self._mw.pulser_mw_ch_SpinBox.value()
        self._pe_logic.apd1_gt_ch = self._mw.apd1_gt_ch_SpinBox.value()
        self._pe_logic.apd2_gt_ch = self._mw.apd2_gt_ch_SpinBox.value()
        self._pe_logic.pulser_rf_freq0 = self._mw.pulser_rf_freq0_DSpinBox.value() * 1e6
        self._pe_logic.pulser_rf_amp0 = self._mw.pulser_rf_amp0_DSpinBox.value()
        self._pe_logic.pulser_mw_length = self._mw.pulser_mw_length_DSpinBox.value() / 1e6
        self._pe_logic.pulser_laser_ch = self._mw.pulser_laser_ch_SpinBox.value()

        # delay parameter:
        self._pe_logic.laser_delay = self._mw.delay_laser_DSpinBox.value()
        self._pe_logic.micro_delay = self._mw.delay_microwave_DSpinBox.value()
        self._pe_logic.gate_sg_delay = self._mw.delay_gate_sig_DSpinBox.value()
        self._pe_logic.gate_rf_delay = self._mw.delay_gate_ref_DSpinBox.value()

        # measurement parameter
        self._pe_logic.current_meas_asset_name = self._mw.current_meas_asset_name_ComboBox.currentText()
        self._pe_logic.tempscan_plot_type = self._mw.tempscan_plot_type_ComboBox.currentText()
        self._pe_logic.integration_time = self._mw.integration_time_DSpinBox.value() / 1e3
        self._pe_logic.x_axis_start = self._mw.x_axis_start_DSpinBox.value() / 1e9
        self._pe_logic.x_axis_stop = self._mw.x_axis_stop_DSpinBox.value() / 1e9
        self._pe_logic.x_axis_stepwidth = self._mw.x_axis_stepwidth_DSpinBox.value() / 1e9
        self._pe_logic.num_of_meas_runs = self._mw.num_of_meas_runs_SpinBox.value()
        self._pe_logic.dD_dT = self._mw.dD_dT_DSpinBox.value() * 1e3
        self._pe_logic.T_0 = self._mw.T_0_SpinBox.value()
        self._pe_logic.num_of_meas_average = self._mw.num_of_meas_average_SpinBox.value()

        # mw parameters for measurement
        self._pe_logic.mw_cw_freq = self._mw.mw_cw_freq_DSpinBox.value() * 1e9
        self._pe_logic.mw_cw_power = self._mw.mw_cw_power_DSpinBox.value()

        # temperature measurement frequencies
        self._pe_logic.t_meas_frq_1 = self._mw.t_meas_frq_1_DSpinBox.value() * 1e9
        self._pe_logic.t_meas_frq_2 = self._mw.t_meas_frq_2_DSpinBox.value() * 1e9
        self._pe_logic.t_meas_frq_3 = self._mw.t_meas_frq_3_DSpinBox.value() * 1e9
        self._pe_logic.t_meas_frq_4 = self._mw.t_meas_frq_4_DSpinBox.value() * 1e9
        self._pe_logic.t_meas_frq_5 = self._mw.t_meas_frq_5_DSpinBox.value() * 1e9
        self._pe_logic.t_meas_frq_6 = self._mw.t_meas_frq_6_DSpinBox.value() * 1e9

    def save_measurement(self):
        """ Save the current measurement.

        @return:
        """
        timestamp = datetime.datetime.now()
        filetag = self._mw.save_tag_LineEdit.text()
        filepath = self._save_logic.get_path_for_module(module_name='PulsedExperiments')

        if len(filetag) > 0:
            filename = os.path.join(filepath, '{0}_{1}_PulsedExp'.format(timestamp.strftime('%Y%m%d-%H%M-%S'), filetag))
        else:
            filename = os.path.join(filepath, '{0}_PulsedExp'.format(timestamp.strftime('%Y%m%d-%H%M-%S'), ))

        exporter_graph = pg.exporters.SVGExporter(self._mw.pulsed_exp_GraphicsView.plotItem.scene())

        # exporter_graph = pg.exporters.ImageExporter(self._mw.odmr_PlotWidget.plotItem)
        exporter_graph.export(filename + '.svg')

        # self._save_logic.
        self._pe_logic.save_pulsed_experiments_measurement(name_tag=filetag, timestamp=timestamp)

    def switch_all_on(self):
        """
        Switch on all PulseStreamer outputs
        """
        self._pe_logic.switch_all_on()

    def switch_all_off(self):
        """
        Switch off all PulseStreamer outputs
        """
        self._pe_logic.switch_all_off()

    def update_meas_graph(self):
        """ Retrieve from the logic the current x and y values and display them
            in the graph.
        """
        if self._pe_logic.current_meas_asset_name in ['Temperature_Scan']:
            if self._mw.tempscan_plot_type_ComboBox.currentText() == 'Temperature':
                # Plot the temperature data
                self.pulsed_exp_graph.setData(self._pe_logic.time_axis_list, self._pe_logic.temp_axis_list)

                # Empty the fit plot
                self.fit_image.setData((), ())

            elif self._mw.tempscan_plot_type_ComboBox.currentText() == 'Normalized intensity':
                if self._pe_logic.fit_result is not None:
                    # Plot the fit that is used in the logic module
                    fit_x_data = np.linspace(self._pe_logic.x_axis_list.min(), self._pe_logic.x_axis_list.max(), 100)
                    fit_y_data = self._pe_logic.fit_result.model.eval(self._pe_logic.fit_result.params, x=fit_x_data)
                    self.fit_image.setData(fit_x_data, fit_y_data)

                else:
                    # Empty the fit plot
                    self.fit_image.setData((), ())

                # Plot the intensity data
                self.pulsed_exp_graph.setData(self._pe_logic.x_axis_list, self._pe_logic.y_axis_list)
        else:
            # Plot the intensity data
            self.pulsed_exp_graph.setData(self._pe_logic.x_axis_list, self._pe_logic.y_axis_list)

            # Empty the fit plot
            self.fit_image.setData((), ())

    def update_meas_parameter(self):
        """ Update the display parameter close to the graph. """

        self._mw.current_meas_index_SpinBox.setValue(self._pe_logic.current_meas_index)
        self._mw.elapsed_time_DSpinBox.setValue(self._pe_logic.elapsed_time)
        self._mw.num_of_current_meas_runs_SpinBox.setValue(self._pe_logic.num_of_current_meas_runs)

        measurement_name = self._pe_logic.current_meas_asset_name
        if measurement_name in ['Pulsed_Rabi']:
            self._mw.current_meas_point_DSpinBox.setValue(self._pe_logic.current_meas_point * 1e6)
        else:
            pass
