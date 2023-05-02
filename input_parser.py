# readline_test.py
f = open("./cpu3_b2.csv", 'r')
#print_format = "python3 dla_mbconv_runner_arg.py --config 112,3,1,32,8,16 >> result.txt"
print_format = "python3 dla_mbconv_runner_arg.py --config {} >> result.txt"
# print("in_s,k,s,in_c,exp_c,out_c,macs,params,latency")
line = f.readline()
while True:
    line = f.readline()
    if not line: break
    input_shape= ",".join(line.split(",")[:6])
    print(print_format.format(input_shape))

f.close()