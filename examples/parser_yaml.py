import numpy as np
from ruamel.yaml import YAML
import argparse

eps = np.finfo(float).eps

# Config running.py files from terminal
# Example, type python ./examples/parser_yaml.py -c test.yaml script1 to run
parser = argparse.ArgumentParser(
    description="Parser Example",
    add_help=True
)
parser.add_argument('-c', '--config_file', help='Specify config file', metavar='FILE')
subparser = parser.add_subparsers(dest='mode')
subparser.add_parser('script1')
subparser.add_parser('script2')

arg = parser.parse_args() # get all arguments in the parser
arg_dict = vars(arg)
for key, value in arg_dict.items():
    print(f'{key:25s} -> {value}')

if arg_dict['mode'] == 'script1':
    print('Running script1')
    # write to .yaml file
    data = {
        'Name': 'Khanh',
        'Age': 22,
        'Subject':[
            1,
            2,
        ],
        'Project':{
            'DOA Estimator':{ 
                'years': '1st'
            },
            'Micromouse': {
                'years': '2nd'
            },
        }
    }

    yaml = YAML()
    yaml.default_flow_style=False
    yaml.indent(mapping = 2, sequence = 4, offset = 2) # dict = map, list/array = sequence
    with open('./test.yaml', 'w') as f:
        yaml.dump(data, f)












