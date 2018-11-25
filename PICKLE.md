# Pickled File Format

## `_processed.p`
"{dataset}_{model}_processed.p" stores a pickled tuple (X, X_processed, Y).

X is the training data.
X_processed is the training data after evaluating the model up to the first
flat layer. (Will have dimension [50000, 512] for cifar datasets and vgg model).
Y holds the labels as integers.

{dataset} (either cifar10 or cifar100) is the source of the training data.

{model} (either cifar10 or cifar100) is the pretrained vgg model to use.
Both these models require loading an h5 file.

## `separated.p`
"{dataset}_{model}_separated.p" stores a pickled dictionary (label_str => (X, X_processed)).

The dictionary maps string labels to a tuple (X, X_processed), where X is an array containing
every image that has that label. And X_processed is an array containing every processed
images that has that label.
