# @Time   : 2022/3/11
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

import argparse

from recbole_cdr.quick_start import run_recbole_cdr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CMF', help='name of models')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_cdr(model=args.model, config_file_list=config_file_list)

"""The code starts by importing argparse, which is a library that allows the user to create command line arguments. Next, it defines an argument parser object and sets its type to be str.
 The default value for this argument is CMF, which stands for "Computer Music Foundation". Next, it defines two arguments: --model and --config_files.
 These are both required arguments because they have no default values. The next step in the code is to parse these two arguments using argparse's parse_known_args() method.
 This will return a list of tuples containing all of the possible options that can be passed into the program as well as their corresponding values if any were provided during parsing (e.g., model=CMF).
 If there are no such options or values provided then None will be returned instead (e.g., config_file_list=None).
 After parsing has been completed, args contains a list of tuples with all of the available options and their corresponding values if any were provided during parsing (e.g., model='CMF').
 The code attempts to parse the arguments for a Python script."""