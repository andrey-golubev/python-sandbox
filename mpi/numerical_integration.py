import math
from mpi4py import MPI
import sys

debug = False


def func(point):
    return 4.0 / (1.0 + math.pow(point, 2))


def step(jobs_count):
    return 0.001 * (1.0 / jobs_count)


# expecting 0-based job numbers
def borders(job_number, jobs_count):
    interval_len = 1.0 / jobs_count
    return job_number * interval_len, (job_number+1) * interval_len



size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

first, last = borders(rank, size)
if debug:
    sys.stdout.write(
        "Process %d gives range: %s.\n"
        % (rank, str(borders(rank, size))))

st = step(size)
if debug:
    sys.stdout.write(
        "Process %d gives step: %f.\n"
        % (rank, st))

partial_integral = 0.0

i = first
while i <= last:
    center = i + st / 2.0
    partial_integral += func(center) * st
    i += st

if rank == 0:
    partial_integral += MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE)
    sys.stdout.write(
        "Function value: %f.\n"
        % partial_integral)
else:
    MPI.COMM_WORLD.send(partial_integral, dest=0)
