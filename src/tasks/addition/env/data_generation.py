"""
data_generation.py

Core script for generating training/test addition data. First, generates random pairs of numbers,
then steps through an execution trace, computing the exact order of subroutines that need to be
called.

"""
import json
import numpy as np
from trace import Trace
import config as addition_config
import os
def generate_addition(prefix, data_dir, num_examples, debug=False, maximum=10000000000, debug_every=1000):
    """
    Generates addition data with the given string prefix (i.e. 'train', 'test') and the specified
    number of examples.

    :param prefix: String prefix for saving the file ('train', 'test')
    :param num_examples: Number of examples to generate.

    """
    in_data = []
    trace_data = []
    for i in range(num_examples):
        in1 = np.random.randint(maximum - 1)
        in2 = np.random.randint(maximum - in1)
        if debug and i % debug_every == 0:
            trace = Trace(in1, in2, True).trace
        else:
            trace = Trace(in1, in2).trace
        in_data.append((in1, in2))
        trace_data.append(trace)

    with open(os.path.join(data_dir, '{}.json'.format(prefix+"_int")), 'w') as f:
        json.dump(in_data, f)
    with open(os.path.join(data_dir, '{}.json'.format(prefix+"_trace")), 'w') as f:
        json.dump(trace_data, f)


if __name__ == '__main__':
    num_training = 4096
    TRAINING_INT_DATA_PATH = os.path.join(addition_config.DATA_DIR, "exp1_" + str(num_training) + "_int.json")
    TRAINING_TRACE_DATA_PATH = os.path.join(addition_config.DATA_DIR, "exp1_" + str(num_training) + "_trace.json")

    # int numbers
    generate_addition('exp1_'+ str(num_training), addition_config.DATA_DIR, num_training, debug=True)
    with open(TRAINING_INT_DATA_PATH, 'r') as f:
        training_int_data = json.load(f)
        print(json.dumps(training_int_data, indent=4, sort_keys=True))
    # [
    #     [
    #         4461781839,
    #         3642465889
    #     ]
    # ]


    # traces
    with open(TRAINING_TRACE_DATA_PATH, 'r') as f:
        training_trace_data = json.load(f)
        print(json.dumps(training_trace_data, indent=4, sort_keys=True))
    # [
    #     [
    #         [
    #             [
    #                 "ADD",
    #                 2
    #             ],
    #             [],
    #             false
    #         ],
    #         [
    #             [
    #                 "ADD1",
    #                 3
    #             ],
    #             [],
    #             false
    #         ],
    #         [
    #             [
    #                 "WRITE",
    #                 1
    #             ],
    #             [
    #                 0,
    #                 8
    #             ],
    #             false
    #         ],
