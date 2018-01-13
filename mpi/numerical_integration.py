import math
from mpi4py import MPI
import sys


def func(point):
    return 4.0 / (1.0 + math.pow(point, 2))


def step(jobs_count):
    return 0.001 * (1.0 / jobs_count)


# expecting 0-based job numbers
def borders(job_number, jobs_count):
    interval_len = 1.0 / jobs_count
    return job_number * interval_len, (job_number+1) * interval_len


if __name__ == "__main__":
    debug = False

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    first, last = borders(rank, size)
    if debug:
        sys.stdout.write(
            "Range #%d: %s.\n"
            % (rank, str(borders(rank, size))))

    st = step(size)
    if debug:
        sys.stdout.write(
            "Step #%d: %f.\n"
            % (rank, st))

    partial_integral = 0.0

    i = first
    while i <= last:
        center = i + st / 2.0
        partial_integral += func(center) * st
        i += st

    if debug:
        sys.stdout.write(
            "Value #%d: %f.\n"
            % (rank, partial_integral))

    if rank == 0:
        for i in range(0, size-1):
            partial_integral += comm.recv(source=MPI.ANY_SOURCE)
        sys.stdout.write(
            "Function value: %f.\n"
            % partial_integral)
    else:
        comm.send(partial_integral, dest=0)
