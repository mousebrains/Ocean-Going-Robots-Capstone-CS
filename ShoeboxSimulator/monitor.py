#! /usr/bin/env python3
#
# This script is connected to two serial ports:
#   One talking to the TWR Shoebox Simulator flight computer
#   The other to a port on the science computer which proglet.dat
#       is configured as an SVS603 device.
#
# The flight computer is monitored and
#   the mission is started via control-P initially.
#   When the flight computer indicates it has reached the surface,
#       it is told to resume the dive immedidiatly.
#
# The SVS603 port is listened to and replicates the expected protocol.
#
# Fall-2020, Pat Welch, pat@mousebrains.com

import serial
import argparse
import logging
import logging.handlers
import threading
import queue
import re
import random

def mkLogger(fn:str) -> logging.Logger:
    logger = logging.getLogger()
    if fn is None:
        ch = logging.StreamHandler()
    else:
        ch = logging.handlers.RotatingFileHandler(fn, maxBytes=10000000, backupCount=2)

    logger.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(threadName)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

class MyCommon(threading.Thread):
    def __init__(self, name:str, port:str, baud:int, logger:logging.Logger, err:queue.Queue):
        threading.Thread.__init__(self, daemon=True)
        self.name = name
        self.port = port
        self.baud = baud
        self.logger = logger
        self.err = err

    def run(self) -> None: # Called on start
        try:
            with serial.Serial(self.port, self.baud) as ser:
                self.__doit(ser)
        except Exception as e:
            self.err.put(e)

    def __doit(self, ser:serial.Serial) -> None:
        line = bytearray()
        while True:
            c = ser.read()
            if c is None:
                raise Exception("EOF on {}".format(self.name))
            if (c == b"\n") or (c == b"\r"):
                if len(line) > 0:
                    try:
                        line = str(line, 'utf-8') # Try and convert to a string
                    except:
                        pass
                    self.logger.info("%s", line)
                    try:
                        self.procLine(line, ser)
                    except Exception as e:
                        self.logger.exception("Error calling procLine")
                        raise e

                line = bytearray()
            else:
                line += c

class Flight(MyCommon):
    def __init__(self, port:str, baud:int, logger:logging.Logger, err:queue.Queue):
        MyCommon.__init__(self, "FLT", port, baud, logger, err)
        self.__resume = re.compile(r"^\s+Hit Control-R to RESUME the mission,.*$")
        self.__start = re.compile(r"\s+The control-P character immediately starts the mission.\s*$")

    def procLine(self, line:str, ser:serial.Serial) -> None:
        if self.__resume.match(line):
            self.logger.info("Sent control-R")
            ser.write(b'\x12') # Control-R is 0x12
            ser.flush()
        elif self.__start.match(line):
            self.logger.info("Sent control-P")
            ser.write(b'\x10') # Control-P is 0x10
            ser.flush()

class SVS(MyCommon):
    def __init__(self, port:str, baud:int, logger:logging.Logger, err:queue.Queue):
        MyCommon.__init__(self, "SVS", port, baud, logger, err)
        self.__setTime = re.compile(r"set time \d+\.\d*")
        self.__restart = re.compile(r"restart")

    def __writeTo(self, ser:serial.Serial, line:str) -> None:
        line = bytes(line + "\r", "utf-8")
        self.logger.info("Sent %s", line)
        ser.write(line)
        ser.flush()

    def procLine(self, line:str, ser:serial.Serial) -> None:
        if self.__setTime.match(line):
            self.__writeTo(ser, "TIME SET:")
            return
        if not self.__restart.match(line):
            return

        fields = []
        for n in range(13):
            fields.append(str(random.random()))

        self.__writeTo(ser, ",".join(fields))

        fields[11] = str(0)
        self.__writeTo(ser, ",".join(fields))

parser = argparse.ArgumentParser()
parser.add_argument("--flt", type=str, default="/dev/tty.usbserial-FTVNDFAK3",
        help="Flight computer serial port")
parser.add_argument("--fltBaud", type=int, default=115200, help="Flight baudrate")
parser.add_argument("--svs", type=str, default="/dev/tty.usbserial-FTVNDFAK1",
        help="SVS603 serial port")
parser.add_argument("--svsBaud", type=int, default=9600, help="SVS603 baudrate")
parser.add_argument("--log", type=str, help="Logfile name")
args = parser.parse_args()

logger = mkLogger(args.log)
logger.info("Args=%s", args)

q = queue.Queue()
flt = Flight(args.flt, args.fltBaud, logger, q)
flt.start()
svs = SVS(args.svs, args.svsBaud, logger, q)
svs.start()

e = q.get()
q.task_done()
logger.exception("Unexpected exception, %s", e)
