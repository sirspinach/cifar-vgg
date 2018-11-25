import keras.datasets
from cifar100vgg import cifar100vgg
from cifar10vgg import cifar10vgg
import pickle
import numpy as np

# Joint script ==> I want my own file.
def main(dataset_name="cifar100", model_name="cifar100"):
    if dataset_name == "cifar100":
        dataset = keras.datasets.cifar100
        labels = CIFAR100_LABELS
    elif dataset_name == "cifar10":
        dataset = keras.datasets.cifar10
        labels = CIFAR10_LABELS
    else:
        raise ValueError(dataset_name)


    print("Loaded dataset={}".format(dataset_name))
    if model_name == "cifar100":
        model = cifar100vgg(train=False)
    elif model_name == "cifar10":
        model = cifar10vgg(train=False)
    else:
        raise ValueError(model_name)
    print("Loaded model={}".format(model_name))

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train_norm = model.normalize_production(x_train.astype('float32'))

    if dataset_name == model_name:
        print("Check validation error:")
        check_validation(dataset_name, x_test, y_test, model)

    print("evaluating first flat layer using dataset...")
    inter_model = keras.Model(inputs=model.model.input,
            outputs=model.model.get_layer("flatten_1").output)
    X = x_train
    Y = y_train
    X_processed = inter_model.predict(x_train_norm)
    data = (X, X_processed, Y)
    print("done.")

    print("saving processed images...")
    filename = "{}_{}_processed.p".format(dataset_name, model_name)
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print("saved results to {}".format(filename))

    print("saving a separated version of the data...")
    # dataset is a map (label_str => (X, X_processed))
    dataset = {}
    for i, label in enumerate(labels):
        ii = (Y.flatten() == i)
        imgs = X[ii]
        imgs_processed = X_processed[ii]
        dataset[label] = (imgs, imgs_processed)

    filename = "{}_{}_separated.p".format(dataset_name, model_name)
    with open(filename, "wb") as f:
        pickle.dump(dataset, f)
    print("saved results to {}".format(filename))


    # Finally, clear the session.
    keras.backend.clear_session()


def check_validation(dataset_name, x_test, y_test, model):
    if dataset_name == "cifar10":
        n_labels = 10
        print("We expect loss={}".format(0.0641))
    elif dataset_name == "cifar100":
        n_labels = 100
        print("We expect loss={}".format(0.2952))
    else:
        raise ValueError(dataset_name)

    y_test_cat = keras.utils.to_categorical(y_test, n_labels)
    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test_cat,1)

    loss = sum(residuals)/len(residuals)
    # Just a sanity check. For cifar100, expect loss=0.2952.
    print("the validation 0/1 loss is: ",loss)

CIFAR10_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog",
            "frog", "horse", "ship", "truck"]
CIFAR100_LABELS = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


for data in ["cifar10", "cifar100"]:
    for model in ["cifar10", "cifar100"]:
        main(dataset_name=data, model_name=model)
