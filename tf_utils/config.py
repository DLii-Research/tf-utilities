import argparse

def create_config(argv=[], arg_defs=[], description=""):
    parser = argparse.ArgumentParser(description=description)
    for fn in arg_defs:
        fn(parser)
    return parser.parse_args(argv)

def config_factory(base_arg_defs=[], description=""):
    def create_config_from_base(argv=[], arg_defs=[], description=description):
        return create_config(argv, base_arg_defs + arg_defs)
    return create_config_from_base