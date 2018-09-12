import os
import time
try:
    import fcntl
except ImportError:
    fcntl = None


class FileLockException(Exception):
    pass


class FileLock(object):
    def __init__(self, file_name, timeout=10, delay=.05, raise_error=True):
        self.is_locked = False
        self.lockfile = os.path.join("%s.lock" % file_name)
        self.file_name = file_name
        self.timeout = timeout
        self.delay = delay
        self.raise_error = raise_error
        self.fd = None

    def acquire(self):
        """Acqire a file lock
        """
        start_time = time.time()
        while True:
            try:
                self.fd = open(self.lockfile, 'w+')
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.is_locked = True
                break
            except (IOError, OSError) as e:
                if (time.time() - start_time) >= self.timeout:
                    if self.raise_error:
                        raise FileLockException("Timeout occured: {}.".format(e))
                    return
                time.sleep(self.delay)

    def release(self):
        """Release the filelock
        """
        if not self.is_locked:
            return
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()
        self.is_locked = False

    def __enter__(self):
        """Entering a with statement
        """
        if not self.is_locked:
            self.acquire()
        return self

    def __exit__(self, t, value, traceback):
        """For the exit of a with statement
        """
        if self.is_locked:
            self.release()

    def __del__(self):
        """Make sure the lock and file does not remain
        """
        self.release()
        try:
            os.remove(self.lockfile)
        except OSError:
            pass


class FakeFileLock(FileLock):
    def acquire(self):
        """Fake aquiring a lock
        """
        pass
