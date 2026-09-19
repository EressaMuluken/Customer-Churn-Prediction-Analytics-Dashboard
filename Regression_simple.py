import numpy as np 
import matplotlib.pyplot as plt 
xs = np.array([1,2,3,4,5,6,7,8])
ys = np.array([2.9,3.4,4.9,4.7,6.2,6.9,7.3,8.6])
lr = 0.01
epochs = 200
loss = []
w = 0
b = 0
for epoch in range(epochs): 
  pred = w * xs + b
  error = pred - ys 
  loss.append(np.mean(error ** 2 / len(xs)))
  dl_dw = 0
  dl_db = 0
  for (index, item) in enumerate(error): 
    dl_dw +=  item * xs[index] / len(xs)
    dl_db +=  item / len(xs)
  w -= lr * dl_dw
  b -= lr * dl_db
plt.plot(xs, ys, 'o')
plt.plot(xs, w*xs + b, '-')
plt.show()