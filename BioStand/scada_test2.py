#!/usr/bin/env python
import minimalmodbus


from serial.tools import list_ports
available_ports = list_ports.comports()
print(f'available ports: {[x.device for x in available_ports]}')

instrument = minimalmodbus.Instrument('COM8', 1)

print(instrument.serial.port)
instrument.serial.baudrate = 9600
instrument.serial.bytesize = 8
instrument.serial.stopbits = 1
instrument.serial.timeout  = 0.05

instrument.address     # this is the slave address number
instrument.mode = minimalmodbus.MODE_RTU   # rtu or ascii mode


while True:
        t1 = instrument.read_register(11, 3)*100 # Registernumber, number of decimals
        print(t1)
        pass
