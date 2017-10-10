import datetime
import sys
import json

class Logger(object):
    """
    A logging that doesn't leave logs open between writes, so as to allow AFS synchronization.
    """

    # Level constants
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __init__(self, log_path=None, json_log_path=None, min_print_level=0, min_file_level=0):
        """
        log_path: The full path for the log file to write. The file will be appended to if it exists.
        min_print_level: Only messages with level above this level will be printed to stderr.
        min_file_level: Only messages with level above this level will be written to disk.
        """
        self.log_path = log_path
        self.json_log_path = json_log_path
        self.min_print_level = min_print_level
        self.min_file_level = min_file_level

    def Log(self, message, level=INFO):
        if level >= self.min_print_level:
            # Write to STDERR
            sys.stderr.write("[%i] %s\n" % (level, message))
        if self.log_path and level >= self.min_file_level:
            # Write to the log file then close it
            with open(self.log_path, 'a') as f:
                datetime_string = datetime.datetime.now().strftime(
                    "%y-%m-%d %H:%M:%S")
                f.write("%s [%i] %s\n" % (datetime_string, level, message))

    def LogJSON(self, message_obj, level=INFO):
        if self.json_log_path and level >= self.min_file_level:
            with open(self.json_log_path, 'w') as f:
                print >>f, json.dumps(message_obj)
        else:
            sys.stderr.write('WARNING: No JSON log filename.')

