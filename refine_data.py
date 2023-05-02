import json

# readline_test.py
f = open("./result_refined.txt", 'r')
print("in_s,k,s,in_c,exp_c,out_c,macs,params,latency")
while True:
    line = f.readline()
    if not line: break
    input_shape, macs, info, params = line.split("|")
    input_shape = list(map(int, input_shape[1:-1].split(", ")))
    macs = int(macs.split(" ")[-1])
    params = int(params.split(" ")[-1])
    info = json.loads(info)
    #print(*input_shape, macs, params, info["mean"], sep=",")
    print(*input_shape, macs, params, info["median"], sep=",")

f.close()