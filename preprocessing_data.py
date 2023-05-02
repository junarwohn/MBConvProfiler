import json

# readline_test.py
f = open("./result_big_case.txt", 'r')
print("in_s,k,s,in_c,exp_c,out_c,macs,params,latency")
while True:
    line = f.readline()
    if not line: break
    input_shape = f.readline()
    params = f.readline()
    macs = f.readline()
    for i in range(4):
        f.readline()
    info = f.readline()

    input_shape = list(map(int, input_shape.split("(")[-1].split(")")[0].split(", ")))
    macs = int(macs.split(" ")[-1])
    params = int(params.split(" ")[-1])
    info = json.loads(info.replace('\'', '\"'))
    print(*input_shape, macs, params, info["mean"], sep=",")
    #print(*input_shape, macs, params, info["median"], sep=",")

f.close()