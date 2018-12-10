"""
config.py

Configuration Variables for the Addition NPI Task => Stores Scratch-Pad Dimensions, Vector/Program
Embedding Information, etc.

"""
import numpy as np
import torch
import os
import sys
real_path = os.path.split(os.path.realpath(__file__))[0]
DATA_DIR = os.path.join(real_path, "../data")

CONFIG = {
    "ENVIRONMENT_ROW": 4,         # Input 1, Input 2, Carry, Output
    "ENVIRONMENT_COL": 10,        # 10-Digit Maximum for Addition Task
    "ENVIRONMENT_DEPTH": 10,      # Size of each element vector => One-Hot, Options: 0-9

    "ARGUMENT_NUM": 2,            # Maximum Number of Program Arguments
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
REV_PROGRAM_ID = {i: x[0] for i, x in enumerate(PROGRAM_SET)}
PADDING = {'ret': 2, 'prog_id': -1, 'args': [0, 0]}


class Arguments():             # Program Arguments
    def __init__(self, args, num_args=CONFIG["ARGUMENT_NUM"], arg_depth=CONFIG["ARGUMENT_DEPTH"]):
        self.args = args
        self.arg_vec = torch.zeros((num_args, arg_depth))