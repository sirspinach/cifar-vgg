import pickle


def inspect(dataset_name="cifar100", model_name="cifar100"):
    filename = "{}_{}_separated.p".format(dataset_name, model_name)
    with open(filename, "rb") as f:
        dataset = pickle.load(f)
    return dataset


if __name__ == '__main__':
    data = inspect()
    img = data['boy'][0][0]
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.savefig("boy.png")
    print("image saved")
