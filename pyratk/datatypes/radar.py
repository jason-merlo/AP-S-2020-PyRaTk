# -*- coding: utf-8 -*-
"""
Radar datatypes

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
from collections import namedtuple
from dataclasses import dataclass
from pyratk.datatypes.geometry import Point

TransmitterTuple = namedtuple('Transmitter', ['location', 'pulses'])
ReceiverTuple = namedtuple('Receiver', ['daq_index', 'location'])

# Detection = namedtuple('Detection', ['location', 'power', 'velocity', 'doppler'])

@dataclass
class Detection:
    '''Class for keeping track of an item in inventory.'''
    location: Point = Point()
    power: float = 0
    velocity: Point = Point()
    doppler: float = 0
