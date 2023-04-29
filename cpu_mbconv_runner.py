#TVM

from mbconv_profiler import MBProfiler

mbprof = MBProfiler('firefly', 'cpu', opt_level=3)
mbprof.do_exec((112,3,2,16,96,24))      #(input_size, kernel_size, stride, input_channel, expanded_channel, output_channel)

print(mbprof.exec_record)