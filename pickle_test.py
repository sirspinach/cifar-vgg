import pickle
import numpy as np

def process2():
    with open("cifar10processed.p", "rb") as f:
        X, X_processed, Y = pickle.load(f)

    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog",
            "frog", "horse", "ship", "truck"]

    dataset = {}
    for i, label in enumerate(labels):
        ii = (Y.flatten() == i)
        imgs = X[ii]
        imgs_processed = X_processed[ii]
        dataset[label] = (imgs, imgs_processed)

    with open("cifar10separated.p", "wb") as f:
        pickle.dump(dataset, f)


with open("cifar10separated.p", "rb") as f:
    dataset = pickle.load(f)
