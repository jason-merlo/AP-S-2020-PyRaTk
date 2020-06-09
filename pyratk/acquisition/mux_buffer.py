# -*- coding: utf-8 -*-
"""
Mux Buffer Class.

Base mux type for data manager.

Author: Jason Merlo
"""

from pyqtgraph import QtCore
from pyratk.formatting import warning

class MuxBuffer(QtCore.QObject):
    """
    MuxBuffer class - base class inherited by DataManager class.

    Handles multiple data input sources and aggrigates them into one data
    "mux" which can select from the various input sources added.
    """
    data_available_signal = QtCore.pyqtSignal(tuple)
    source_reset_signal = QtCore.pyqtSignal()

    def __init__(self):
        """Initialize MuxBuffer Class."""
        super().__init__()
        self.source_list = []
        self.source = None

    # === SOURCE ==============================================================

    def add_source(self, source):
        self.source_list.append(source)
        source.data_available_signal.connect(self.data_available_signal.emit)
        source.reset_signal.connect(self.reset_signal.emit)

    def set_source(self, source):
        # Pause current source
        if self.source:
            self.source.close()
        # Switch to new source, adding if necessisary
        if source not in self.source_list:
            self.add_source(source)
        self.source = source

        # warning(self.source_list)

    def get_source(self):
        return self.source

    def get_samples(self):
        self.source.get_samples()

    # === PROPERTIES ==========================================================

    @property
    def sample_rate(self):
        return self.source.sample_rate

    @property
    def sample_chunk_size(self):
        return self.source.sample_chunk_size

    @property
    def daq_type(self):
        return self.source.daq_type

    @property
    def type(self):
        return self.source.type


    @property
    def paused(self):
        return self.source.paused

    @paused.setter
    def paused(self, p):
        self.source.paused = p
