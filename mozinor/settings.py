# -*- coding: utf-8 -*-
"""
Created on July 2017

@author: JulienWuthrich
"""
import os
import time
import logging

from mozinor.loggs import LogFile


DEBUG = True

ABSOLUTE_PATH = os.path.abspath(os.path.dirname(__file__))
LOG_ROOT = os.path.join(ABSOLUTE_PATH, 'logs')

now = time.strftime("%Y%m%d%H%M%S", time.gmtime())
logfile = ("_".join(["Mozinor", now, ".log"]))
logpath = os.path.join(LOG_ROOT, logfile)
logger = LogFile(logpath, logging.DEBUG, DEBUG)
