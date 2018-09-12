import os
import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

_comm = None
MPI_RANK = 0
MPI_SIZE = 1
if MPI:
    _comm = MPI.COMM_WORLD
    MPI_RANK = _comm.Get_rank()
    MPI_SIZE = _comm.Get_size()


def get_mpi_name():
    """Get the machien name
    :rtype: str
    """
    if MPI:
        name = MPI.Get_processor_name()
    else:
        import socket
        name = socket.gethostname()
    return "{}({}/{})".format(name, MPI_RANK, MPI_SIZE)


def scatter(send_buf, chunk_size, dtype):
    """Scatter numpy array
    :param send_buf: array to scatter
    :type send_buf: np.ndarray
    :param chunk_size: size of the received buffer
    :param dtype: data type of numpy array
    """
    recv_buf = np.empty(chunk_size, dtype=dtype)
    _comm.Scatter(send_buf, recv_buf, root=0)
    return recv_buf


def get_job_config(config=None):
    """Get the global job configuration
    :param: config at rank 0
    :return: a synchronized dictionary between jobs
    :rtype: dict
    """
    if MPI_RANK != 0:
        config = None
    return _comm.bcast(config, root=0)


def barrier():
    """Add a comm barrier
    """
    _comm.Barrier()


def main():
    """Initial sync between jobs
    """
    if os.environ.get('OMPI_COMM_WORLD_SIZE') or 1 <= 1:
        return
    # wait for all the workers to start
    print("{}: barrier for workers {}/{}".format(
        get_mpi_name(), os.environ.get('OMPI_COMM_WORLD_RANKS'), os.environ.get('OMPI_COMM_WORLD_SIZE'))
    )
    barrier()


if __name__ == '__main__':
    main()
