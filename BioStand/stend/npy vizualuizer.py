import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def load_npy_file():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("NPY files", "*.npy")])
    if file_path:
        data = np.load(file_path)
        plt.imshow(data, cmap='gray')
        plt.axis('off')
        plt.show()

load_npy_file()
