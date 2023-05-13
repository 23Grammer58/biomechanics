import numpy as np
from PIL import Image
def gray():
    path = 'C:/Users/Lenovo/PycharmProjects/MainBioStand/BioStand/files/GoreTexPulleyTest/grayscale_npy/0.npy'
    arr = np.load(path)
    arr = arr.astype(np.uint8)
    print(arr.shape)
    print(arr)
    np.save("test.npy",arr )
    img = Image.fromarray(arr)
    img.show()
def f():
    path = 'C:/Users/Lenovo/PycharmProjects/MainBioStand/BioStand/files/GoreTexPulleyTest/Frame/1681818632.85942.npy'
    arr = np.load(path)
    print(arr.shape)
    print(arr)

gray()
f()