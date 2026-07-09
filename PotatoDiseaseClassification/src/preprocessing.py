import os
import cv2
import matplotlib.pyplot as plt


def show_samples(base_path):

    classes = sorted(
    folder
    for folder in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, folder))
)

    plt.figure(figsize=(15,5))

    for i, cls in enumerate(classes):

        folder = os.path.join(base_path, cls)

        image_name = os.listdir(folder)[0]

        image_path = os.path.join(folder, image_name)

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.subplot(1,3,i+1)

        plt.imshow(image)

        plt.title(cls)

        plt.axis("off")

    plt.show()