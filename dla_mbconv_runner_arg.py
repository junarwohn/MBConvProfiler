from mbconv_profiler import MBProfiler
import argparse

parser = argparse.ArgumentParser(description='dla_mbconv_runner_arg')

# 입력받을 인자값 등록
parser.add_argument('--config', required=True, help='어느 것을 요구하냐')
args = parser.parse_args()


config = args.config
config = tuple(map(int, config.split(",")))


mbprof = MBProfiler('xavier', 'dla', opt_level=3)
#mbprof.do_exec((112,3,2,16,96,24))      #(input_size, kernel_size, stride, input_channel, expanded_channel, output_channel)
mbprof.do_exec(config)      #(input_size, kernel_size, stride, input_channel, expanded_channel, output_channel)

print(mbprof.exec_record)