import argparse
import sys
import os.path
from dectree.transpiler import transpile, CONFIG_DEFAULTS


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(prog=__package__,
                                     description="Generates a Python module in directory OUTPUT_DIR "
                                                 "from each decision tree given in SOURCE_FILE. If OUTPUT_DIR "
                                                 "is not given, Python modules are created next to their "
                                                 "SOURCE_FILE.")
    parser.add_argument("src",
                        metavar='SOURCE_FILE',
                        nargs='+',
                        help="source file containing a decision tree (YAML format)")
    parser.add_argument("-o", "--out",
                        metavar='OUTPUT_DIR',
                        help="target directory for generated Python files")

    for option_name, option_def in CONFIG_DEFAULTS.items():
        default, help = option_def
        parser.add_argument('--' + option_name,
                            default=default,
                            action='store_true' if isinstance(default, bool) else None,
                            help=help.format(default=default))

    args = parser.parse_args(args=args)
    options = {k: v for k, v in vars(args).items() if k in CONFIG_DEFAULTS}

    if args.no_jit and args.vectorize:
        print('warning: --vectorize has no effect because of --no_jit')

    out_dir = args.out
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    for src_file in args.src:
        out_file = None
        if out_dir is not None:
            basename = os.path.splitext(os.path.basename(src_file))[0] + '.py'
            out_file = os.path.join(out_dir, basename)
        out_file = transpile(src_file, out_file=out_file, **options)
        print('generated', out_file)


if __name__ == '__main__':
    main()
