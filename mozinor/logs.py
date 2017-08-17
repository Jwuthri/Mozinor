# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import logging


class LogFile(object):
    """Class to create and use logs."""

    def __init__(self, logfile, level="INFO", show=False, fmt="%(message)s"):
        """Create a logger.

            Args:
            -----
                logfile (str): the name of the logfile
                level (logging.level): the level of log (DEBUG, INFO, WARNING)
                    by default INFO
                show (bool): show log in the console
                    by default False
                fmt (str): format of the logs
                    by default "%(message)s"
        """
        self.logfile = logfile
        self.level = level
        self.fmt = fmt
        self.logger = logging.getLogger(logfile)
        self.logger.setLevel(level)
        self.hfile()
        if show:
            self.hstream()

    def hfile(self):
        """Handler file.

            Return:
            -------
                logging.FileHandler
        """
        hdlr = logging.FileHandler(self.logfile, encoding="utf-8")
        hdlr.setLevel(self.level)
        hdlr.setFormatter(logging.Formatter(self.fmt))
        self.logger.addHandler(hdlr)

    def hstream(self):
        """Handler stream.

            Return:
            -------
                logging.StreamHandler
        """
        hdlr = logging.StreamHandler()
        hdlr.setLevel(self.level)
        hdlr.setFormatter(logging.Formatter(self.fmt))
        self.logger.addHandler(hdlr)

    def log(self, msg, level="INFO"):
        """Write in the logfile.

            Args:
            -----
                msg (str): message to write
                level (logging.level): type of level

            Return:
            -------
                Logfile update with the new message
        """
        self.logger.log(level, msg)

    def kill(self):
        """Delete all the handlers.

            Return:
            -------
                Remove all handler
        """
        for hdlr in self.logger.handlers:
            self.logger.removeHandler(hdlr)
