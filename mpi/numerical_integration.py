import math
from mpi4py import MPI
import sys

import parsing_utils as pu


def func(point):
    return 4.0 / (1.0 + math.pow(point, 2))


def step(jobs_count):
    return 0.001 * (1.0 / jobs_count)


# expecting 0-based job numbers
def borders(job_number, jobs_count):
    interval_len = 1.0 / jobs_count
    return job_number * interval_len, (job_number+1) * interval_len


class IntegralCalculator:
    stub_value = 0.0

    @staticmethod
    def calculate(_method_name, _i, _step):
        _methods = {
            "riemann": IntegralCalculator._riemann_sum,
            "trapezoid": IntegralCalculator._trapezoid_rule,
            "simpson": IntegralCalculator._simpson_rule,
            "gauss": IntegralCalculator._gaussian_quadrature
        }
        return _methods.get(_method_name, IntegralCalculator._stub)(_i, _step)

    @staticmethod
    def _riemann_sum(_i, _step):
        _center = _i + _step / 2.0
        return func(_center) * _step

    @staticmethod
    def _trapezoid_rule(_i, _step):
        return _step * (func(_i) + func(_i + _step)) / 2.0

    @staticmethod
    def _simpson_rule(_i, _step):
        _end = _i + _step
        _center = _i + _step / 2.0
        return (func(_i) + 4.0 * func(_center) + func(_end)) * (_end - _i) / 6.0

    @staticmethod
    def _gaussian_quadrature(_i, _step):
        _end = _i + _step
        _center = _i + _step / 2.0
        _some_point = (_end - _i) / (2 * math.sqrt(3))
        return (func(_center - _some_point) + func(_center + _some_point)) * (_end - _i) / 2.0

    @staticmethod
    def _stub(_i, _step):
        return IntegralCalculator.stub_value


if __name__ == "__main__":
    args = pu.parser_instance().parse_args()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if args.debug and rank == 0:
        sys.stdout.write("Args: %s\n" % str(args.__dict__))

    first, last = borders(rank, size)
    if args.debug:
        sys.stdout.write(
            "Range #%d: %s\n"
            % (rank, str(borders(rank, size))))

    st = args.step if args.step != -1.0 else step(size)
    if args.debug:
        sys.stdout.write(
            "Step #%d: %f\n"
            % (rank, st))

    partial_integral = 0.0

    i = first
    while i <= last:
        partial_integral += IntegralCalculator.calculate(args.method, i, st)
        i += st

    if args.debug:
        sys.stdout.write(
            "Value #%d: %f\n"
            % (rank, partial_integral))

    if rank == 0:
        for i in range(0, size-1):
            partial_integral += comm.recv(source=MPI.ANY_SOURCE)
        sys.stdout.write(
            "Function value: %f\n"
            % partial_integral)
    else:
        comm.send(partial_integral, dest=0)
