import argparse
import csv
import shlex
import subprocess
import time
from functools import reduce

parser = argparse.ArgumentParser(description='')
parser.add_argument('--jobs', default=[2], nargs='+',
                    help='List of jobs amount to run\n'
                         'Example: 2 3 4 5 6\n'
                         'Result: executing the program withing 2, 3, 4, 5 and 6 jobs separately')
parser.add_argument('--methods', default=['riemann'], nargs='+',
                    help='List of all method to run\n'
                         'Possible values: [riemann, trapezoid, simpson, gauss]')
parser.add_argument('--times', default=100, type=int,
                    help='How many times to repeat each execution')
parser.add_argument('--step', type=float, default=-1.0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--out_csv_file', default="out.csv", type=str)


def get_avg(lst, n):
    return reduce((lambda a, b: a + b), lst) / n


if __name__ == "__main__":
    args = parser.parse_args()

    debug_str = '--debug' if args.debug else ''

    exec_outputs = []
    for num_jobs in args.jobs:
        for method in args.methods:
            num_jobs = int(num_jobs)
            command = f"mpiexec -n {num_jobs} python3.6 numerical_integration.py " \
                      f"--method={method} --step={args.step} {debug_str}"
            cmd_args = shlex.split(command)

            elapsed_list = []
            values = []
            error = ""
            for i in range(args.times):
                start = time.time()
                p = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                p.wait()
                elapsed_list.append(time.time() - start)

                output, error = p.communicate()
                output = output.decode('ascii')
                if error:
                    print("Error during run: ", error)
                    error = "Internal Error"
                    continue
                out_value = float(''.join(c for c in output if c.isdigit() or c == '.'))
                values.append(out_value)
                pass
            exec_outputs.append({
                'error': error,
                'step': args.step,
                'method': method,
                'jobs': num_jobs,
                'avg_time': get_avg(elapsed_list, args.times),
                'avg_value': get_avg(values, args.times)
            })

    with open(args.out_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Error', 'Step', 'Method', 'Job Number', 'Avg. Time', 'Avg. Value'])
        for out in exec_outputs:
            out_list = [value for key, value in out.items()]
            writer.writerow(out_list)
            print(out_list)
