# Importing dependencies
from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


def poison_images(number):
    # Defining a poisoning backdoor attack
    backdoor = PoisoningAttackBackdoor(perturbation=add_pattern_bd)

    # (train_images, train_labels), (test_images, test_labels), min, max = load_dataset(name="mnist")

    # Defining a target label for poisoning
    target = to_categorical(
        labels=np.repeat(a=5, repeats=number),
        nb_classes=10
    )

    # Inspecting the target labels
    # print(f"The target labels for poisoning are\n {target}")

    # Poisoning sample data
    poisoned_images, poisoned_labels = backdoor.poison(
        x=train_images[:number],
        y=target
    )

    poisoned_labels_numerical = np.argmax(poisoned_labels, axis=1)

    # # Creating a figure and axes for the poisoned images
    # fig, axes = plt.subplots(
    #     nrows=1,
    #     ncols=5,
    #     squeeze=True,
    #     figsize=(15, 5)
    #     )
    #
    # # Plotting the poisoned images
    # for i in range(len(poisoned_images)):
    #     axes[i].imshow(X=poisoned_images[i])
    #     axes[i].set_title(label=f"Label: {np.argmax(poisoned_labels[i])}")
    #     axes[i].set_xticks(ticks=[])
    #     axes[i].set_yticks(ticks=[])
    #
    # # Showing the plot
    # plt.show()

    return poisoned_images, poisoned_labels_numerical
