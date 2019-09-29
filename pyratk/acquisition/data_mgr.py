# -*- coding: utf-8 -*-
"""
Data Manager Class.

Controls data source selection.

Author: Jason Merlo
Maintainer: Jason Merlo (merlojas@msu.edu)
"""
import h5py                     # Used for hdf5 database
from pyratk.acquisition.mux_buffer import MuxBuffer
from pyratk.acquisition.virtual_daq import VirtualDAQ
from pyratk.acquisition import daq   # Extention of DAQ object
from pyqtgraph import QtCore


# COMPRESSION OPTIONS
COMPRESSION = "gzip"
COMPRESSION_OPTS = 9


class DataManager(MuxBuffer):
    """
    DataManager class; extends MuxBuffer class.

    Handles multiple data input sources and aggrigates them into one data
    "mux" which can select from the various input sources added.
    """
    reset_signal = QtCore.pyqtSignal()

    def __init__(self, db="default.hdf5", daq=None):
        """
        Initialize DataManager Class.

        Arguments:
            db (optional)
                database to save/load from
        """
        super().__init__()

        self.db = None
        self.samples = None     # Pointer to the samples dataset
        self.labels = None      # Pointer to the labels dataset

        # Open virtual daq for playback
        self.virt_daq = VirtualDAQ()
        self.add_source(self.virt_daq)
        self.daq = daq

        self.source_reset_signal.connect(self.reset_signal.emit)

        # DEVEL/DEBUG
        try:
            self.open_database(db)
        except Exception as e:
            print("(DataManager) Error opening debug database:", e)
            raise RuntimeError("Cannot open specified dataset - is it already"
                               " opened elsewhere?")

    # === DATABASE ============================================================

    def open_database(self, db_path, mode='a'):
        """
        Accept file path of a database to load and open it.

        db_path: string
            Path to file to open
        mode: string
            File access mode (default: 'a')
        """
        # If a database is open, close it then open a new one
        if self.db is not None:
            self.db.close()

        # Attempt to open or create a new file with the specified name
        try:
            self.db = h5py.File(db_path, mode)
        except OSError as e:
            print("(DataManager) Error opening HDF5 File:", e)
            raise RuntimeError("Cannot open or create specified dataset - is "
                               "it already opened elsewhere?")

        # Attempt to open 'samples', 'labels', 'subjects' datasets
        # Create new datsets on keyerror
        try:
            self.samples = self.db["samples"]
        except KeyError:
            print("No 'samples' group found. Creating 'samples'...")
            self.samples = self.db.create_group("samples")
        except Exception as e:
            # Handle all other exceptions
            print(e)

        try:
            self.labels = self.db["labels"]
        except KeyError:
            print("No 'labels' group found. Creating 'labels'")
            self.labels = self.db.create_group("labels")
        except Exception as e:
            # Handle all other exceptions
            print(e)

        try:
            self.subjects = self.db["subjects"]
        except KeyError:
            print("No 'subjects' group found. Creating 'subjects'")
            self.subjects = self.db.create_group("subjects")
        except Exception as e:
            # Handle all other exceptions
            print(e)

    # === DATASET CONTROL =====================================================

    def load_dataset(self, ds):
        """Load dataset into virtualDAQ and set to virtualDAQ source."""
        # self.reset()
        self.virt_daq.load_dataset(ds)
        self.set_source(self.virt_daq)
        self.reset()

    def get_datasets(self):
        """Return list of all dataset objects in 'samples' dataset."""
        keys = []
        if self.db:
            for key in self.samples:
                keys.append(self.samples[key])
            return keys
        else:
            print("(DataManager) Database must be loaded before datasets read")

    def delete_dataset(self, ds):
        """Remove dataset from database."""
        try:
            del self.db[ds.name]
        except Exception as e:
            print("Error deleting dataset: ", e)

    def save_buffer(self, name, labels, subject, notes):
        """
        Write buffer contents to dataset with specified 'name'.

        If no name is provided, 'sample_n' will be used, where n is the index
        of the sample relative to 0.
        """
        if "/samples/{:}".format(name) in self.db:
            ds = self.samples[name]
        else:
            # Does not exist, create new entry
            try:
                # Save buffer data
                ds = self.samples.create_dataset(
                    name, data=self.source.ts_buffer,
                    compression=COMPRESSION,
                    compression_opts=COMPRESSION_OPTS)
            except Exception as e:
                print("(DataManager) Error saving dataset: ", e)

        # Create hard links to class label group and subject group
        if labels is None:
            labels_str = ''
        else:
            labels_str = []  # used for saving labels to attribute
            for label in labels:
                # Get type of label
                if type(label) is str:
                    label_name = label
                else:
                    label_name = label.name
                # print('Checking if',
                #       "labels/{:}/{:}".format(label_name, name), 'exists')
                if "{:}/{:}".format(label_name, name) not in self.labels:
                    self.labels[label_name][name] = ds
                labels_str.append(label_name.split('/')[-1])
            labels_str = ','.join(labels_str)

        if subject:
            if type(subject) is str:
                subject_name = subject
            else:
                subject_name = subject.name
            if "{:}/{:}".format(subject_name, name) not in self.subjects:
                self.subjects[subject_name][name] = ds

        if notes is None:
            notes = ''

        try:
            # Save attribute data
            attrs = self.samples[name].attrs
            attrs.create("sample_rate", self.source.sample_rate)
            attrs.create("sample_size", self.source.sample_chunk_size)
            attrs.create("daq_type", self.source.daq_type.encode('utf8'))
            attrs.create("num_channels", self.source.num_channels)
            attrs.create("label", labels_str.encode('utf8'))
            if subject:
                if type(subject) is str:
                    subject_name = subject
                else:
                    subject_name = subject.name
                attrs.create("subject",
                             subject_name.split('/')[-1].encode('utf8'))
            attrs.create("notes", notes.encode('utf8'))
        except Exception as e:
            print("(DataManager) Error saving attributes: ", e)

    def remove_attributes(self, name, labels, subject):
        """
        Remove hard-linked created in attribute folders.

        Currently the only hard-linked attribues are: labels, subject
        """
        # Attempt to open dataset
        if "/samples/{:}".format(name) in self.db:
            ds = self.samples[name]
        else:
            print("(DataManager) Error attempting to remove attributes.")
            print("(DataManager) Dataset does not exist.")

        # Attempt to delete sample from label dataset
        for label in labels:
            try:
                self.labels[label].pop(name)
            except KeyError:
                print("(DataManager) Error attempting to remove sample \
                       hard-link in label dataset.")
                print("(DataManager) sample does not exist in dataset.")
                print("Name: {:}".format(name))

        # Attempt to delete sample from subjects dataset
        if subject:
            try:
                self.subjects[subject].pop(name)
            except KeyError:
                print("(DataManager) Error attempting to remove sample \
                       hard-link in subject dataset.")
                print("(DataManager) sample does not exist in dataset.")
                print("Name: {:}".format(name))

    # --- labels --- #
    def get_labels(self):
        """Return list of all label objects in 'labels' dataset."""
        keys = []
        if self.db:
            for key in self.labels:
                keys.append(self.labels[key])
            return keys
        else:
            print("(DataManager) Database must be loaded before datasets read")

    def add_label(self, label):
        """Add a label to dataset."""
        try:
            self.labels.create_group(label)
        except Exception as e:
            # Handle all exceptions
            print('(DataManager)', e)

    def remove_label(self):
        """TODO: implement remove_label."""
        pass

    # --- subjects --- #
    def get_subjects(self):
        """Return list of all subject objects in dataset."""
        keys = []
        if self.db:
            for key in self.subjects:
                keys.append(self.subjects[key])
            return keys
        else:
            print("(DataManager) Database must be loaded before datasets read")

    def add_subject(self, subject):
        """Add a subject to dataset."""
        try:
            self.subjects.create_group(subject)
        except Exception as e:
            # Handle all exceptions
            print('(DataManager)', e)

        def remove_subject():
            # TODO: implement remove_subject
            pass

    # === DATA CONTROL ========================================================

    def reset(self):
        """Reset DAQ manager, clear all data and graphs."""
        self.source.paused = True
        self.reset_signal.emit()
        self.source.reset()
        self.source.paused = False

    def pause_toggle(self):
        """Pauses the DAQ manager."""
        # Virtual DAQ needs a dataset loaded before running
        if self.source is not self.virt_daq or self.virt_daq.ds is not None:
            if self.source.paused is True:
                # self.reset()  # used for arbitrary dt
                self.source.paused = False
            else:
                self.source.paused = True

    def close(self):
        """Close the selected object in the DAQ manager."""
        for source in self.source_list:
            source.close()

    # === PROPERTIES ====================================================

    # @property
    # def reset_flag(self):
    #     return self.source.reset_flag
    #
    # @reset_flag.setter
    # def reset_flag(self, a):
    #     self.source.reset_flag = a

    @property
    def ts_buffer(self):
        return self.source.ts_buffer
