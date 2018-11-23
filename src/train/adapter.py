import json


# element of input_trace is dict={'ret':B*T*1,'prog_id:B*T*1','args:B*T*x'}
def traceJson_to_traceInput(batchsize, args_dim, padding, in_file, out_file):
  with open(in_file, 'r') as fin:
    # element of traces is list=[[["prog_name",prog_id],args,ret],[...],[...]]
    traces = json.load(fin)
  
  traces_input = []
  padding_token = [[0], [0] * args_dim, 1]
  for i in range(len(traces)):
    if i + batchsize > len(traces):
      break
    trace_input = {'ret': [], 'prog_id': [], 'args': []}
    max_length = 0
    for j in range(batchsize):
      trace_input['ret'].append([])
      trace_input['prog_id'].append([])
      trace_input['args'].append([])
      for step in traces[i + j]:
        trace_input['ret'][j].append([1 if step[2] == True else 0])
        trace_input['prog_id'][j].append([step[0][1]])
        trace_input['args'][j].append(step[1] + [0] * (3 - len(step[1])))
      if len(trace_input['ret'][j]) > max_length:
        max_length = len(trace_input['ret'][j])
    if padding:
      for j in range(batchsize):
        trace_input['ret'][j] += [padding_token[2]] * (max_length - len(trace_input['ret'][j]))
        trace_input['prog_id'][j] += [padding_token[0]] * (max_length - len(trace_input['prog_id'][j]))
        trace_input['args'][j] += [padding_token[1]] * (max_length - len(trace_input['args'][j]))
    traces_input.append(trace_input)
  
  with open(out_file, 'w') as fout:
    json.dump(traces_input, fout)


if __name__ == "__main__":
  traceJson_to_traceInput(batchsize=1, args_dim=3, padding=True,
                          in_file="./src/tasks/addition/data/train_trace.json",
                          out_file="./src/tasks/addition/data/train_trace_input.json")
