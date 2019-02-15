import time
import logging


# no_op method/object that accept every signature
class NoOp(object):
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op


class FileLogger(object):
    def __init__(self, output_dir, is_master=False, is_rank0=False, title="yolo_training"):
        self.output_dir = output_dir
        self.title = title

        # Log to console if rank 0, Log to console and file if master
        if not is_rank0:
            self.logger = NoOp()
        else:
            self.logger = self.get_logger(output_dir, log_to_file=is_master)


    def get_logger(self, output_dir, log_to_file=True):
        logger = logging.getLogger(self.title)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(message)s')

        if log_to_file:
            vlog = logging.FileHandler(output_dir + '/verbose.log')
            vlog.setLevel(logging.INFO)
            vlog.setFormatter(formatter)
            logger.addHandler(vlog)
            eventlog = logging.FileHandler(output_dir + '/event.log')
            eventlog.setLevel(logging.WARN)
            eventlog.setFormatter(formatter)
            logger.addHandler(eventlog)
            time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
            debuglog = logging.FileHandler(output_dir + '/debug.log')
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(time_formatter)

            logger.addHandler(debuglog)

        console = logging.StreamHandler()

        console.setFormatter(formatter)

        console.setLevel(logging.DEBUG)

        logger.addHandler(console)

        return logger

    def console(self, *args):
        self.logger.debug(*args)

    def event(self, *args):
        self.logger.warn(*args)

    def verbose(self, *args):
        self.logger.info(*args)
