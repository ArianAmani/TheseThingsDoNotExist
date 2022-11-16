import os
import gdown
from zipfile import ZipFile
import tensorflow as tf


def download_celebA():
    os.makedirs("celeba")

    url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
    output = "celeba/data.zip"
    gdown.download(url, output, quiet=True)

    with ZipFile("celeba/data.zip", "r") as zipobj:
        zipobj.extractall("celeba")


def create_dataset(batch_size=32, img_size=128, download=True):
    if download:
        download_celebA()

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "celeba",
        label_mode=None,
        validation_split=0.1,
        subset="training",
        seed=1234,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "celeba",
        label_mode=None,
        validation_split=0.1,
        subset="validation",
        seed=1234,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )

    train_dataset = train_dataset.map(
        lambda x: x / 255.0).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(
        lambda x: x / 255.0).prefetch(tf.data.AUTOTUNE)

    return [train_dataset, val_dataset]
