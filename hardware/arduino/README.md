# Arduino hardware modules for Qudi
This README is about the Arduino hardware modules for Qudi.\
It was written by Oliver Gerull in 2021. Feel free to use, contribute to and change this README and the modules.

## Hardware module "arduino_mega2560.py"
To make this Python module work, you have to do the following:

### Use the right hardware
This module has been implemented for the following hardware. \
If you use other hardware, you might have to adapt the code of python module and Arduino sketch.

#### Arduino Mega 2560
The Arduino Mega 2560 is connected to the computer, Qudi is running on, via USB cable.\
Don't connect anything to the Digital Output pins 22-42.

#### PCF8591
The PCF8591 is connected to the Arduino via I2C.\
It controls the VCA (voltage controlled attenuator) that controls the microwave power.

#### DAC8734
The DAC8734 is connected to the Arduino via SPI.\
It controls two piezo stages and the VCO (voltage controlled oscillator) that generates microwaves.

### Upload Arduino firmware sketch
Upload the following sketch to your Arduino Mega2560:\
hardware/arduino/Arduino firmware/FirmataExpress_SPI/FirmataExpress_SPI.ino

### Prepare Qudi's Python environment
Use at least Python3.7 (required by "pymata4").\
Install the Python package "pymata4" in your Python environment.

### Update VCA and VCO calibration data
Do a calibration measurement for your own VCA and VCO hardware and save it to "hardware/arduino/Calibration data".