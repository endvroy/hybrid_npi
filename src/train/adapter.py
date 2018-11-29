import json
import os

# element of input_trace is dict={'ret':T*B*1,'prog_id:T*B*1','args:T*B*x'}
def trace_json_to_input(batchsize, args_dim, padding, in_file):
    # read traces
    with open(in_file, 'r') as fin:
        # element of traces is list=[[["prog_name",prog_id],args,ret],[...],[...]]
        traces = json.load(fin)
    traces_input = []
    padding_token = [0, [0] * args_dim, 1]
    # print("len(traces):", len(traces))
    for i in range(0, len(traces), batchsize):
        if i + batchsize > len(traces):
            # print("break:", i)
            break
        max_length = 0
        for j in range(batchsize):
            if max_length < len(traces[i + j]):
                max_length = len(traces[i + j])
        trace_input = {'ret': [], 'prog_id': [], 'args': [], 'len': max_length}
        trace_input['ret'] = []
        trace_input['prog_id'] = []
        trace_input['args'] = []
        # align traces in batch
        for l in range(max_length):
            ret_input = []
            prog_input = []
            args_input = []
            for j in range(batchsize):
                if len(traces[i + j]) <= l:
                    ret_input.append(padding_token[2])
                    prog_input.append(padding_token[0])
                    args_input.append(padding_token[1])
                else:
                    ret_input.append(1 if traces[i + j][l][2] == True else 0)
                    prog_input.append(traces[i + j][l][0][1])
                    args_input.append(traces[i + j][l][1] + [0] * (args_dim - len(traces[i + j][l][1])))
            trace_input['ret'].append(ret_input)
            trace_input['prog_id'].append(prog_input)
            trace_input['args'].append(args_input)
        traces_input.append(trace_input)
    # print("len(trace_input)", len(traces_input))
    return traces_input

if __name__ == "__main__":
    trace_json_to_input(batchsize=2, args_dim=3, padding=True,
                        in_file="./src/tasks/addition/data/train_trace.json",
                        out_file="./src/tasks/addition/data/train_trace_input.json")
