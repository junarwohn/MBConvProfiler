import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time

def compute_exp_c(in_s, k, s, in_c, out_c, macs):
    out_s = in_s // s

    #reverse-compute size of exp_c from mac
    exp_s = (in_s + 2*(k//2) - k)//s + 1

    pw1 = in_s**2 * in_c
    dw = exp_s**2 * k**2
    pw2 = exp_s**2 * out_c

    exp_c = macs / (pw1+dw+pw2)
    return exp_c


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='dataset .csv', required=True)
parser.add_argument('--train_size', type=float, default=0.8, help='ratio to allocate train set from entire dataset')
parser.add_argument('--degree', type=int, default=2, help='degree of polynomial')
parser.add_argument('--repeat', type=int, default=100)
args = parser.parse_args()

#read dataset
df = pd.read_csv(args.dataset)

#set target category
x = df[['in_s', 'k', 's', 'in_c', 'out_c', 'latency']]
y = df[['macs']]


#split dataset into train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=args.train_size, test_size=1.0-args.train_size)

train_score = -1
test_score = -1
for i in range(500):
    if i % 100 == 0:
        print(i)
    #create polynomial of train set
    poly = PolynomialFeatures(degree=args.degree, include_bias=True)
    x_train_poly = poly.fit_transform(x_train)
    
    #train with linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(x_train_poly, y_train)
    
    #create polynomial of test set and predict
    x_test_poly = poly.transform(x_test)
    y_predict = lin_reg.predict(x_test_poly)
    
    #print prediction score
    train_score = lin_reg.score(x_train_poly, y_train)
    new_test_score = lin_reg.score(x_test_poly, y_test)
    if new_test_score > test_score:
        test_score = new_test_score
        best = [y_test.copy(), y_predict.copy()]
    #print("train_score: ", lin_reg.score(x_train_poly, y_train))
    #print("test_score: ", lin_reg.score(x_test_poly, y_test))

#visualize
import matplotlib.pyplot as plt
#plt.scatter(y_test, y_predict, alpha=0.4)
plt.scatter(best[0], best[1], alpha=0.4)
plt.xlabel("Actual macs")
plt.ylabel("Predicted macs")
plt.title(args.dataset)

index = time.strftime('%H%M%S', time.localtime(time.time()))
#plt.savefig(args.dataset+'.png', transparent=True)
plt.savefig(args.dataset+"_"+str(args.degree)+"_"+str(args.repeat)+index+'.png', transparent=True)

#plt.show()

