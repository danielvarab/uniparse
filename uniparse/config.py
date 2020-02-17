import argparse
from argparse import Namespace
from ast import literal_eval
from configparser import ConfigParser


class ParameterConfig(argparse.Action):
    """Action that injects config file (.ini format) into parser namespace"""

    def __call__(self, parser, namespace, values, option_strings=None):
        options = ConfigParser()
        options.read(values)
        option_pairs = {
            name: literal_eval(value)
            for section in options.sections()
            for name, value in options.items(section)
        }
        # set config to namespace
        setattr(namespace, self.dest, values)

        # add config options to namespace
        for k, v in option_pairs.items():
            setattr(namespace, k, v)
            # parser.add_argument(f"--{k}", default=v, required=False)


def pprint_dict(namespace):
    """
    Adapted from https://github.com/zysite/biaffine-parser/blob/master/parser/config.py
    
    Cheers!
    """
    s = line = "-" * 15 + "-+-" + "-" * 45 + "\n"
    s += f"{'Param':15} | {'Value':^45}\n" + line
    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        s += f"{name:15} | {str(value):^45}\n"
    s += line

    print(s)
