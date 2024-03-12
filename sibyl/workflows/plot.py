import pickle

import matplotlib.pyplot as plt
import numpy as np


def main():
    vl: list = pickle.load(open("variance_loss.pkl", "rb"))
    mse: list = pickle.load(open("sum_loss.pkl", "rb"))

    plt.plot(vl, label="Variance Loss", color="red", alpha=0.5)
    plt.plot(mse, label="MSE Loss", color="blue", alpha=0.5)
    print(np.array(vl).mean())
    print(np.array(mse).mean())
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("loss.png")


if __name__ == "__main__":
    main()
