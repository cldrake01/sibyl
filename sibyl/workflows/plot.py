import pickle

import matplotlib.pyplot as plt
import numpy as np

vl: list = pickle.load(open("vl.pkl", "rb"))
mse: list = pickle.load(open("mse.pkl", "rb"))

plt.plot(vl, label="Variance Loss", color="red", alpha=0.5)
plt.plot(mse, label="MSE Loss", color="blue", alpha=0.5)
print(np.array(vl).mean())
print(np.array(mse).mean())
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig("loss.png")
