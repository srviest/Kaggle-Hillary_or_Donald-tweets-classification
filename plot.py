import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



acc_pd = pd.read_csv('./model_rnn/loss.csv')
c_acc_pd = pd.read_csv('./model_crnn/loss.csv')
results = acc_pd.as_matrix()
c_results = c_acc_pd.as_matrix()


max_len = len(results)

# downsample
downsample = 5
index = [ downsample*i for i in range(max_len/downsample)]
x = np.array(range(max_len))

# '-.', '--', ':', '-'
linestyle='-'
linewidth=1

plt.plot(x[index],results[index], linestyle=linestyle, linewidth=linewidth, label='acc')
plt.plot(x[index],c_results[index], linestyle=linestyle, linewidth=linewidth, label='acc', color='red')
plt.ylabel('Loss')
plt.xlabel('Iteration')
# xmin, xmax = plt.xlim()
# print(xmin, xmax)
# plt.xlim(xmin=0, xmax=1.1)
# ymin, ymax = plt.ylim()
# print(ymin, ymax)
# plt.ylim(ymax=1.1)

# fig = plt.gcf()
# size = fig.get_size_inches()*fig.dpi
# DPI = fig.get_dpi()
# fig.set_size_inches(640/float(DPI),480/float(DPI))
# print(size)

plt.tight_layout()
plt.show()

