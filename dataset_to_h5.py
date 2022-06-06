"""
pip install -q tensorflow==2.0.0
pip install adversarial-robustness-toolbox[all]
pip install h5py==2.10.0
pip install git+https://github.com/JanderHungrige/tf.keras-vggface
pip install Pillow
"""

import h5py
from PIL import Image
import pathlib
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def write_dataset(data, label, filename="test.h5"):
    with h5py.File(filename, 'a') as f:
        f.create_dataset(f"data/{label}", data=data)


def read_image(img_path) -> np.ndarray:
    with Image.open(img_path) as f:
        im = f.resize((224, 224))
        return tf.keras.preprocessing.image.img_to_array(im)


def convert_dataset(data_path="./test", h5path="test.h5"):
    for d in tqdm(tuple(pathlib.Path(data_path).iterdir())):
        data = []
        for path in d.iterdir():
            data.append(read_image(path))
        data = np.array(data)
        write_dataset(data, d.name, h5path)
    return data

h5path="test_0.h5"
data = convert_dataset("./test", h5path=h5path)
