"""
This file contains the Qudi Interfuse that enables you to use a SlowCounter as an InStreamer.
It's derived from the data_instream_dummy and adjusted to its new purpose by Oliver Gerull.

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

from core.module import Base
from core.connector import Connector
from core.configoption import ConfigOption
from core.util.helpers import natural_sort
from interface.data_instream_interface import DataInStreamInterface, DataInStreamConstraints
from interface.data_instream_interface import StreamingMode, StreamChannelType, StreamChannel


class InStreamerSlowCounterInterfuse(Base, DataInStreamInterface):
    """ This file contains the Qudi Interfuse that enables you to use a SlowCounter as an InStreamer.

    Example config for copy-paste:

    instreamer_slowcounter_interfuse:
        module.Class: 'interfuse.instreamer_slowcounter_interfuse.InStreamerSlowCounterInterfuse'
        connect:
            slowcounter: 'niusb_6210_slowcounter'
    """

    # connectors
    slowcounter = Connector(interface='SlowCounterInterface')

    # config options
    _digital_channels = tuple()
    _analog_channels = tuple()
    _digital_event_rates = 100000
    _analog_amplitudes = 10

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        # Internal settings
        self.__sample_rate = -1.0
        self.__data_type = np.float64
        self.__stream_length = -1
        self.__buffer_size = -1
        self.__use_circular_buffer = False
        self.__streaming_mode = None
        self.__active_channels = tuple()

        # Data buffer
        self._data_buffer = np.empty(0, dtype=self.__data_type)
        self._has_overflown = False
        self._is_running = False
        self._last_read = None
        self._start_time = None

        # Stored hardware constraints
        self._constraints = None
        return

    def on_activate(self):
        """
        Initialisation performed during activation of the module.
        """
        self._slowcounter_device = self.slowcounter()

        if not self._slowcounter_device.is_running:
            try:
                self._slowcounter_device.start_measure()
            except:
                self.log.error("The slow counter is not yet running and couldn't be started. Please check what's wrong "
                               "before starting SlowCounting.")
                return -1

        self._digital_channels = self._slowcounter_device.get_scanner_count_channels()

        # Sanity check ConfigOptions
        if not self._digital_channels and not self._analog_channels:
            raise Exception('Not a single analog or digital channel provided in ConfigOptions.')
        self._digital_channels = natural_sort(str(chnl) for chnl in self._digital_channels)

        if self._digital_channels:
            try:
                if len(self._digital_channels) != len(self._digital_event_rates):
                    if len(self._digital_event_rates) == 1:
                        tmp = self._digital_event_rates[0]
                        self._digital_event_rates = [i * tmp for i, _ in
                                                     enumerate(self._digital_channels, 1)]
                    else:
                        raise Exception('ConfigOption "digital_event_rates" must have same length '
                                        'as "digital_channels" or just be a single value.')
            except TypeError:
                self._digital_event_rates = [i * self._digital_event_rates for i, _ in
                                             enumerate(self._digital_channels, 1)]

        # Create constraints
        self._constraints = DataInStreamConstraints()
        self._constraints.digital_channels = tuple(
            StreamChannel(name=ch, type=StreamChannelType.DIGITAL, unit='counts') for ch in
            self._digital_channels)
        self._constraints.analog_channels = tuple(
            StreamChannel(name=ch, type=StreamChannelType.ANALOG, unit='V') for ch in
            self._analog_channels)
        self._constraints.analog_sample_rate.min = 1
        self._constraints.analog_sample_rate.max = 2**31-1
        self._constraints.analog_sample_rate.step = 1
        self._constraints.analog_sample_rate.unit = 'Hz'
        self._constraints.digital_sample_rate.min = 1
        self._constraints.digital_sample_rate.max = 2**31-1
        self._constraints.digital_sample_rate.step = 1
        self._constraints.digital_sample_rate.unit = 'Hz'
        self._constraints.combined_sample_rate = self._constraints.analog_sample_rate

        self._constraints.read_block_size.min = 1
        self._constraints.read_block_size.max = 1000000
        self._constraints.read_block_size.step = 1

        # TODO: Implement FINITE streaming mode
        self._constraints.streaming_modes = (StreamingMode.CONTINUOUS,)  # , StreamingMode.FINITE)
        self._constraints.data_type = np.float64
        self._constraints.allow_circular_buffer = True

        self.__sample_rate = self._constraints.combined_sample_rate.min
        self.__data_type = np.float64
        self.__stream_length = 1
        self.__buffer_size = 1000
        self.__use_circular_buffer = False
        self.__streaming_mode = StreamingMode.CONTINUOUS
        self.__active_channels = tuple()

        # Reset data buffer
        self._data_buffer = np.empty(0, dtype=self.__data_type)
        self._has_overflown = False
        self._is_running = False
        self._last_read = None
        self._start_time = None
        return

    def on_deactivate(self):
        """ Shut down the NI card.
        """
        self._has_overflown = False
        self._is_running = False
        self._last_read = None
        # Free memory if possible while module is inactive
        self._data_buffer = np.empty(0, dtype=self.__data_type)
        return

    @property
    def sample_rate(self):
        """
        Read-only property to return the currently set sample rate

        @return float: current sample rate in Hz
        """
        return self.__sample_rate

    @sample_rate.setter
    def sample_rate(self, rate):
        if self._check_settings_change():
            if not self._clk_frequency_valid(rate):
                if self._analog_channels:
                    min_val = self._constraints.combined_sample_rate.min
                    max_val = self._constraints.combined_sample_rate.max
                else:
                    min_val = self._constraints.digital_sample_rate.min
                    max_val = self._constraints.digital_sample_rate.max
                self.log.warning(
                    'Sample rate requested ({0:.3e}Hz) is out of bounds. Please choose '
                    'a value between {1:.3e}Hz and {2:.3e}Hz. Value will be clipped to '
                    'the closest boundary.'.format(rate, min_val, max_val))
                rate = max(min(max_val, rate), min_val)
            self.__sample_rate = float(rate)
        return

    @property
    def data_type(self):
        """
        Read-only property to return the currently set data type

        @return type: current data type
        """
        return self.__data_type

    @property
    def buffer_size(self):
        """
        Read-only property to return the currently buffer size.
        Buffer size corresponds to the number of samples per channel that can be buffered. So the
        actual buffer size in bytes can be estimated by:
            buffer_size * number_of_channels * size_in_bytes(data_type)

        @return int: current buffer size in samples per channel
        """
        return self.__buffer_size

    @buffer_size.setter
    def buffer_size(self, size):
        if self._check_settings_change():
            size = int(size)
            if size < 1:
                self.log.error('Buffer size smaller than 1 makes no sense. Tried to set {0} as '
                               'buffer size and failed.'.format(size))
                return
            self.__buffer_size = int(size)
            self._init_buffer()
        return

    @property
    def use_circular_buffer(self):
        """
        Read-only property to return a flag indicating if circular sample buffering is being used
        or not.

        @return bool: indicate if circular sample buffering is used (True) or not (False)
        """
        return self.__use_circular_buffer

    @use_circular_buffer.setter
    def use_circular_buffer(self, flag):
        if self._check_settings_change():
            if flag and not self._constraints.allow_circular_buffer:
                self.log.error('Circular buffer not allowed for this hardware module.')
                return
            self.__use_circular_buffer = bool(flag)
        return

    @property
    def streaming_mode(self):
        """
        Read-only property to return the currently configured streaming mode Enum.

        @return StreamingMode: Finite (StreamingMode.FINITE) or continuous
                               (StreamingMode.CONTINUOUS) data acquisition
        """
        return self.__streaming_mode

    @streaming_mode.setter
    def streaming_mode(self, mode):
        if self._check_settings_change():
            mode = StreamingMode(mode)
            if mode not in self._constraints.streaming_modes:
                self.log.error('Unknown streaming mode "{0}" encountered.\nValid modes are: {1}.'
                               ''.format(mode, self._constraints.streaming_modes))
                return
            self.__streaming_mode = mode
        return

    @property
    def number_of_channels(self):
        """
        Read-only property to return the currently configured number of data channels.

        @return int: the currently set number of channels
        """
        return len(self._slowcounter_device.get_scanner_count_channels())

    @property
    def active_channels(self):
        """
        The currently configured data channel properties.
        Returns a dict with channel names as keys and corresponding StreamChannel instances as
        values.

        @return dict: currently active data channel properties with keys being the channel names
                      and values being the corresponding StreamChannel instances.
        """
        constr = self._constraints
        return (*(ch.copy() for ch in constr.digital_channels if ch.name in self.__active_channels),
                *(ch.copy() for ch in constr.analog_channels if ch.name in self.__active_channels))

    @active_channels.setter
    def active_channels(self, channels):
        if self._check_settings_change():
            channels = tuple(channels)
            avail_chnl_names = tuple(ch.name for ch in self.available_channels)
            if any(ch not in avail_chnl_names for ch in channels):
                self.log.error('Invalid channel to stream from encountered: {0}.\nValid channels '
                               'are: {1}'
                               ''.format(channels, avail_chnl_names))
                return
            self.__active_channels = channels
        return

    @property
    def available_channels(self):
        """
        Read-only property to return the currently used data channel properties.
        Returns a dict with channel names as keys and corresponding StreamChannel instances as
        values.

        @return tuple: data channel properties for all available channels with keys being the
                       channel names and values being the corresponding StreamChannel instances.
        """
        return (*(ch.copy() for ch in self._constraints.digital_channels),
                *(ch.copy() for ch in self._constraints.analog_channels))

    @property
    def available_samples(self):
        """
        Read-only property to return the currently available number of samples per channel ready
        to read from buffer.

        @return int: Number of available samples per channel
        """
        if not self.is_running:
            return 0
        return 1

    @property
    def stream_length(self):
        """
        Property holding the total number of samples per channel to be acquired by this stream.
        This number is only relevant if the streaming mode is set to StreamingMode.FINITE.

        @return int: The number of samples to acquire per channel. Ignored for continuous streaming.
        """
        return self.__stream_length

    @stream_length.setter
    def stream_length(self, length):
        if self._check_settings_change():
            length = int(length)
            if length < 1:
                self.log.error('Stream_length must be a positive integer >= 1.')
                return
            self.__stream_length = length
        return

    @property
    def is_running(self):
        """
        Read-only flag indicating if the data acquisition is running.

        @return bool: Data acquisition is running (True) or not (False)
        """
        return self._is_running

    @property
    def buffer_overflown(self):
        """
        Read-only flag to check if the read buffer has overflown.
        In case of a circular buffer it indicates data loss.
        In case of a non-circular buffer the data acquisition should have stopped if this flag is
        coming up.
        Flag will only be reset after starting a new data acquisition.

        @return bool: Flag indicates if buffer has overflown (True) or not (False)
        """
        return self._has_overflown

    @property
    def all_settings(self):
        """
        Read-only property to return a dict containing all current settings and values that can be
        configured using the method "configure". Basically returns the same as "configure".

        @return dict: Dictionary containing all configurable settings
        """
        return {'sample_rate': self.__sample_rate,
                'streaming_mode': self.__streaming_mode,
                'active_channels': self.active_channels,
                'stream_length': self.__stream_length,
                'buffer_size': self.__buffer_size,
                'use_circular_buffer': self.__use_circular_buffer}

    def configure(self, sample_rate=None, streaming_mode=None, active_channels=None,
                  stream_length=None, buffer_size=None, use_circular_buffer=None):
        """
        Method to configure all possible settings of the data input stream.

        @param float sample_rate: The sample rate in Hz at which data points are acquired
        @param StreamingMode streaming_mode: The streaming mode to use (finite or continuous)
        @param iterable active_channels: Iterable of channel names (str) to be read from.
        @param int stream_length: In case of a finite data stream, the total number of
                                            samples to read per channel
        @param int buffer_size: The size of the data buffer to pre-allocate in samples per channel
        @param bool use_circular_buffer: Use circular buffering (True) or stop upon buffer overflow
                                         (False)

        @return dict: All current settings in a dict. Keywords are the same as kwarg names.
        """
        if self._check_settings_change():
            # Handle sample rate change
            if sample_rate is not None:
                self.sample_rate = sample_rate

            # Handle streaming mode change
            if streaming_mode is not None:
                self.streaming_mode = streaming_mode

            # Handle active channels
            if active_channels is not None:
                self.active_channels = active_channels

            # Handle total number of samples
            if stream_length is not None:
                self.stream_length = stream_length

            # Handle buffer size
            if buffer_size is not None:
                self.buffer_size = buffer_size

            # Handle circular buffer flag
            if use_circular_buffer is not None:
                self.use_circular_buffer = use_circular_buffer
        return self.all_settings

    def get_constraints(self):
        """
        Return the constraints on the settings for this data streamer.

        @return DataInStreamConstraints: Instance of DataInStreamConstraints containing constraints
        """
        return self._constraints.copy()

    def start_stream(self):
        """
        Start the data acquisition and data stream.

        @return int: error code (0: OK, -1: Error)
        """
        if self.is_running:
            self.log.warning('Unable to start input stream. It is already running.')
            return 0

        self._init_buffer()
        self._is_running = True
        self._start_time = time.perf_counter()
        self._last_read = self._start_time
        return 0

    def stop_stream(self):
        """
        Stop the data acquisition and data stream.

        @return int: error code (0: OK, -1: Error)
        """
        if self.is_running:
            self._is_running = False
        return 0

    def read_data_into_buffer(self, buffer, number_of_samples=None):
        """
        Read data from the stream buffer into a 1D/2D numpy array given as parameter.
        In case of a single data channel the numpy array can be either 1D or 2D. In case of more
        channels the array must be 2D with the first index corresponding to the channel number and
        the second index serving as sample index:
            buffer.shape == (self.number_of_channels, number_of_samples)
        The numpy array must have the same data type as self.data_type.
        If number_of_samples is omitted it will be derived from buffer.shape[1]

        This method will not return until all requested samples have been read or a timeout occurs.

        @param numpy.ndarray buffer: The numpy array to write the samples to
        @param int number_of_samples: optional, number of samples to read per channel. If omitted,
                                      this number will be derived from buffer axis 1 size.

        @return int: Number of samples read into buffer; negative value indicates error
                     (e.g. read timeout)
        """

        return 0

    def read_available_data_into_buffer(self, buffer):
        """
        Read data from the stream buffer into a 1D/2D numpy array given as parameter.
        In case of a single data channel the numpy array can be either 1D or 2D. In case of more
        channels the array must be 2D with the first index corresponding to the channel number and
        the second index serving as sample index:
            buffer.shape == (self.number_of_channels, number_of_samples)
        The numpy array must have the same data type as self.data_type.

        This method will read all currently available samples into buffer. If number of available
        samples exceed buffer size, read only as many samples as fit into the buffer.

        @param numpy.ndarray buffer: The numpy array to write the samples to

        @return int: Number of samples read into buffer; negative value indicates error
                     (e.g. read timeout)
        """
        if not self.is_running:
            self.log.error('Unable to read data. Device is not running.')
            return -1

        avail_samples = min(buffer.size // self.number_of_channels, self.available_samples)
        return self.read_data_into_buffer(buffer=buffer, number_of_samples=avail_samples)

    def read_data(self, number_of_samples=None):
        """
        Read data from the stream buffer into a 2D numpy array and return it.
        The arrays first index corresponds to the channel number and the second index serves as
        sample index:
            return_array.shape == (self.number_of_channels, number_of_samples)
        The numpy arrays data type is the one defined in self.data_type.
        If number_of_samples is omitted all currently available samples are read from buffer.

        This method will not return until all requested samples have been read or a timeout occurs.

        @param int number_of_samples: optional, number of samples to read per channel. If omitted,
                                      all available samples are read from buffer.

        @return numpy.ndarray: The read samples
        """
        if not self.is_running:
            self.log.error('Unable to read data. Device is not running.')
            return np.empty((0, 0), dtype=self.data_type)

        if number_of_samples is None:
            count_data = self._slowcounter_device.read_count_diff(1 / self.__sample_rate)

        else:
            count_data = np.zeros([number_of_samples, self.number_of_channels], dtype='float64')
            for sample in range(number_of_samples):
                count_data[sample] = self._slowcounter_device.read_count_diff(1 / self.__sample_rate)

        # InStreamer give counts/integration_time, not counts/s
        count_data = count_data/self.__sample_rate

        # A new dimension has to be added. Otherwise the array won't be transposed in the next step.
        if count_data.ndim == 1:
            count_data = count_data[np.newaxis]
        # Transpose Array because InStreamer logic needs it that way
        count_data = count_data.transpose()

        return count_data

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
        if not self.is_running:
            self.log.error('Unable to read data. Device is not running.')
            return np.empty(0, dtype=self.__data_type)

        return self._slowcounter_device.read_single_point()

    # =============================================================================================
    def _clk_frequency_valid(self, frequency):
        if self._analog_channels:
            max_rate = self._constraints.combined_sample_rate.max
            min_rate = self._constraints.combined_sample_rate.min
        else:
            max_rate = self._constraints.digital_sample_rate.max
            min_rate = self._constraints.digital_sample_rate.min
        return min_rate <= frequency <= max_rate

    def _init_buffer(self):
        if not self.is_running:
            self._data_buffer = np.zeros(
                self.number_of_channels * self.buffer_size,
                dtype=self.data_type)
            self._has_overflown = False
        return

    def _check_settings_change(self):
        """
        Helper method to check if streamer settings can be changed, i.e. if the streamer is idle.
        Throw a warning if the streamer is running.

        @return bool: Flag indicating if settings can be changed (True) or not (False)
        """
        if self.is_running:
            self.log.warning('Unable to change streamer settings while streamer is running. '
                             'New settings ignored.')
            return False
        return True
