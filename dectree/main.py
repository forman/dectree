import argparse
import os.path
import sys

from dectree.config import CONFIG_DEFAULTS, VECTORIZE_PROP
from dectree.transpiler import transpile


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog=__package__,
        description="Generates a Python module in directory OUTPUT_DIR"
                    " from each decision tree given in SOURCE_FILE."
                    " If OUTPUT_DIR is not given, Python modules are"
                    " created next to their SOURCE_FILE."
    )
    parser.add_argument(
        "src",
        metavar='SOURCE_FILE',
        nargs='+',
        help="source file containing a decision tree (YAML format)"
    )
    parser.add_argument(
        "-o", "--out",
        metavar='OUTPUT_DIR',
        help="target directory for generated Python files"
    )

    for option_name, option_def in CONFIG_DEFAULTS.items():
        default, help_pattern, choices = option_def
        if choices:
            parser.add_argument(
                '--' + option_name,
                default=default,
                help=help_pattern.format(default=default),
                choices=choices
            )
        else:
            parser.add_argument(
                '--' + option_name,
                default=default,
                action='store_true' if isinstance(default, bool) else None,
                help=help_pattern.format(default=default)
            )

    args = parser.parse_args(args=args)
    options = {k: v
               for k, v in vars(args).items()
               if k in CONFIG_DEFAULTS and v != CONFIG_DEFAULTS[k][0]}

    if args.no_jit and args.vectorize == VECTORIZE_PROP:
        print(f'error: --no_jit is illegal because --vectorize'
              f' "{VECTORIZE_PROP}" requires JIT')
        exit(1)

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
