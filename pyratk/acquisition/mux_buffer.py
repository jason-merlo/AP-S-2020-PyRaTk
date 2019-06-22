# -*- coding: utf-8 -*-
"""
Mux Buffer Class.

Base mux type for data manager.

Author: Jason Merlo
last_modified: 8/21/2018
"""


class MuxBuffer(object):
    """
    MuxBuffer class - base class inherited by DataManager class.

    Handles multiple data input sources and aggrigates them into one data
    "mux" which can select from the various input sources added.
    """

    def __init__(self):
        """Initialize MuxBuffer Class."""
        self.source_list = []

    # === SOURCE ==============================================================

    def add_source(self, source):
        self.source_list.append(source)

    def set_source(self, source):
        if source not in self.source_list:
            self.add_source(source)
        self.source = source

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
    def buffer(self):
        return self.source.buffer

    @property
    def paused(self):
        return self.source.paused

    @paused.setter
    def paused(self, p):
        self.source.paused = p
