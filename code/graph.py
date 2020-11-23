import matplotlib.pyplot as plt
import json
import numpy as np
data_1dcnn_Adam_L2 = np.array(json.load(open("1dcnn_Adam_L2_0.001.json", "r")))[:,0]
data_1dcnn_Adam_no_L2 = np.array(json.load(open("1dcnn_Adam_no_L2.json", "r")))[:, 0]
data_1dcnn_RMSprop_no_L2 = np.array(json.load(open("1dcnn_RMSprop_no_L2.json", "r")))[:, 0]
data_1dcnn_SGD_no_L2 = np.array(json.load(open("1dcnn_SGD_no_L2.json", "r")))[:, 0]
x = list(range(data_1dcnn_Adam_L2.shape[0]))

plt.plot(x, data_1dcnn_Adam_L2, label = "Adam with weight decay 0.001")
plt.plot(x, data_1dcnn_Adam_no_L2, label = "Adam with no weight decay")
plt.plot(x, data_1dcnn_RMSprop_no_L2, label = "RMSprop with no weight decay")
plt.plot(x, data_1dcnn_SGD_no_L2, label = "SGD with no weight decay")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss.png")