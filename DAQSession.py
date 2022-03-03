import time
import numpy as np
from uldaq import (get_daq_device_inventory, DaqDevice, AInScanFlag, ScanStatus,
                   ScanOption, create_float_buffer, InterfaceType, AiInputMode, WaitType,
                   DaqEventType, ULException, EventCallbackArgs)

DAQ_RATE = 50000 # fixed for DAQ Card

class DAQSession(object):
    def __init__(self, scan_duration_seconds, low_channel=0, num_channels=1):
        
        interface_type = InterfaceType.USB
        self.event_types = None
        self.range_index = 0
        self.rate = DAQ_RATE
        self.low_channel = low_channel
        self.high_channel = low_channel + num_channels - 1
        self.samples_per_channel = int(DAQ_RATE * scan_duration_seconds)
        self.scan_options = ScanOption.DEFAULTIO # CONTINUOUS || DEFAULTIO
        self.flags = AInScanFlag.DEFAULT

        try:
            # Get descriptors for all of the available DAQ devices.
            devices = get_daq_device_inventory(interface_type)
            number_of_devices = len(devices)
            if number_of_devices == 0:
                raise Exception('Error: No DAQ devices found')
            print('number of daq_devices:', number_of_devices)

            # Create the DAQ device object associated with the specified descriptor index.
            self.daq_device = DaqDevice(devices[-1])

            # Get the AiDevice object and verify that it is valid.
            ai_device = self.daq_device.get_ai_device()
            if ai_device is None:
                raise Exception('Error: The DAQ device does not support analog input')

            # Verify that the specified device supports hardware pacing for analog input.
            ai_info = ai_device.get_info()
            if not ai_info.has_pacer():
                raise Exception('\nError: The specified DAQ device does not support hardware paced analog input')

            # Establish a connection to the DAQ device.
            self.daq_device.connect()

            # The default input mode is SINGLE_ENDED, else if not supported, set to DIFFERENTIAL.
            self.input_mode = AiInputMode.SINGLE_ENDED
            if ai_info.get_num_chans_by_mode(AiInputMode.SINGLE_ENDED) <= 0:
                self.input_mode = AiInputMode.DIFFERENTIAL

            # Get the number of channels and validate the high channel number.
            number_of_channels = ai_info.get_num_chans_by_mode(self.input_mode)
            if self.high_channel >= number_of_channels:
                self.high_channel = number_of_channels - 1
            channel_count = self.high_channel - self.low_channel + 1

            # Get a list of supported ranges and validate the range index.
            self.ranges = ai_info.get_ranges(self.input_mode)
            if self.range_index >= len(self.ranges):
                self.range_index = len(self.ranges) - 1

            # Allocate a buffer to receive the data.
            self.data = create_float_buffer(channel_count, self.samples_per_channel)
        except Exception as e:
            print('ERROR:: \n', e)

    def getStatus(self):
        if self.daq_device:
            status, transfer_status = self.daq_device.get_ai_device().get_scan_status()  
            return status

    def isIdle(self):
        if self.daq_device:
            return self.getStatus() == ScanStatus.IDLE

    def isRunning(self):
        if self.daq_device:
            return self.getStatus() == ScanStatus.RUNNING

    def startScan(self):
        if self.daq_device:
            actualRate = self.daq_device.get_ai_device().a_in_scan(self.low_channel, self.high_channel, self.input_mode, 
                                        self.ranges[self.range_index], self.samples_per_channel,
                                        DAQ_RATE, self.scan_options, self.flags, self.data)
            return actualRate

    def write_scanned_buffer( self, f_name ):
        rawScan = np.reshape( self.data, (self.samples_per_channel,-1))
        with open(f_name, "a") as f: # should be 'w' is first, 'a' if second
            np.savetxt(f, rawScan)
            f.write('+++\n')

    def enableEvent(self, user_data):
        if self.daq_device:
            self.event_types = (DaqEventType.ON_END_OF_INPUT_SCAN 
                    | DaqEventType.ON_INPUT_SCAN_ERROR
                    | DaqEventType.ON_DATA_AVAILABLE )
            self.daq_device.enable_event(self.event_types, 100, self.event_callback_function, user_data)

    def event_callback_function(self, event_callback_args):
        event_type = event_callback_args.event_type
        event_data = event_callback_args.event_data
        user_data = event_callback_args.user_data

        if event_type == DaqEventType.ON_DATA_AVAILABLE:
            scan_count = event_data
            curr_time_stamp = time.time()
            user_data["timeIdx"][user_data["curr"]] = [curr_time_stamp, scan_count]
            user_data["curr"] += 1

        if event_type == DaqEventType.ON_INPUT_SCAN_ERROR:
            exception = ULException(event_data)
            print(exception)
            user_data["status"]['error'] = True

        if event_type == DaqEventType.ON_END_OF_INPUT_SCAN:
            with open(user_data["f_name"], 'a') as f: # should be w if first, a if second
                np.savetxt(f, user_data["timeIdx"][np.all(user_data["timeIdx"] != 0, axis=1)] )
                f.write('+++\n')
            user_data["status"]['complete'] = True

    def disableEvent(self):
        if self.daq_device:
            self.daq_device.disable_event(self.event_types)

    def close(self):
        if self.daq_device:
            # Stop the acquisition if it is still running.
            if self.isRunning():
                self.DaqDevice.get_ai_device().scan_stop()
            if self.daq_device.is_connected():
                self.daq_device.disconnect()
            self.daq_device.release()
            self.daq_device = False