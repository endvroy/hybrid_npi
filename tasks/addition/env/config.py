"""
config.py

Configuration Variables for the Addition NPI Task => Stores Scratch-Pad Dimensions, Vector/Program
Embedding Information, etc.

"""
import numpy as np


CONFIG = {
    "ENVIRONMENT_ROW": 4,         # Input 1, Input 2, Carry, Output
    "ENVIRONMENT_COL": 10,        # 10-Digit Maximum for Addition Task
    "ENVIRONMENT_DEPTH": 10,      # Size of each element vector => One-Hot, Options: 0-9

    "ARGUMENT_NUM": 3,            # Maximum Number of Program Arguments
    "ARGUMENT_DEPTH": 11,         # Size of Argument Vector => One-Hot, Options 0-9, Default (10)
    "DEFAULT_ARG_VALUE": 10,      # Default Argument Value

    "PROGRAM_NUM": 6,             # Maximum Number of Subroutines
    "PROGRAM_KEY_SIZE": 5,        # Size of the Program Keys
    "PROGRAM_EMBEDDING_SIZE": 10  # Size of the Program Embeddings
}

PROGRAM_SET = [
    ("MOVE_PTR", 4, 2),       # Moves Pointer (4 options) either left or right (2 options)
    ("WRITE", 2, 10),         # Given Carry/Out Pointer (2 options) writes digit (10 options)
    ("ADD",),                 # Top-Level Add Program (calls children routines)
    ("ADD1",),                # Single-Digit (Column) Add Operation
    ("CARRY",),               # Carry Operation
    ("LSHIFT",)               # Shifts all Pointers Left (after Single-Digit Add)
]

PROGRAM_ID = {x[0]: i for i, x in enumerate(PROGRAM_SET)}


class Arguments():             # Program Arguments
    def __init__(self, args, num_args=CONFIG["ARGUMENT_NUM"], arg_depth=CONFIG["ARGUMENT_DEPTH"]):
        self.args = args
        self.arg_vec = np.zeros((num_args, arg_depth), dtype=np.float32)


def get_args(args, arg_in=True):
    if arg_in:
        arg_vec = np.zeros((CONFIG["ARGUMENT_NUM"], CONFIG["ARGUMENT_DEPTH"]), dtype=np.int32)
    else:
        arg_vec = [np.zeros((CONFIG["ARGUMENT_DEPTH"]), dtype=np.int32) for _ in
                   range(CONFIG["ARGUMENT_NUM"])]
    if len(args) > 0:
        for i in range(CONFIG["ARGUMENT_NUM"]):
            if i >= len(args):
                arg_vec[i][CONFIG["DEFAULT_ARG_VALUE"]] = 1
            else:
                arg_vec[i][args[i]] = 1
    else:
        for i in range(CONFIG["ARGUMENT_NUM"]):
            arg_vec[i][CONFIG["DEFAULT_ARG_VALUE"]] = 1
    return arg_vec.flatten() if arg_in else arg_vec