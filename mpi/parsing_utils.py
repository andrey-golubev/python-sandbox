def parser_instance():
    import argparse
    parser = argparse.ArgumentParser(description="Parser for function integral calculation")
    parser.add_argument('--method',
                        type=str,
                        default="riemann",
                        help="Possible values: [riemann, trapezoid, simpson, gauss]")
    parser.add_argument('--step',
                        type=float,
                        default=-1.0)
    parser.add_argument('--debug',
                        action='store_true')
    return parser
