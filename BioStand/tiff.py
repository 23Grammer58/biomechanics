import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QSlider
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PIL import Image

class TIFFViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("TIFF Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.canvas = None
        self.slider = None
        self.current_page = 0

        self.open_button = QPushButton("Открыть .tiff файл")
        self.open_button.clicked.connect(self.open_tiff_file)
        self.layout.addWidget(self.open_button)

    def open_tiff_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("TIFF files (*.tiff *.tif);;All files (*.*)")

        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                tiff_file_path = selected_files[0]
                self.show_tiff_file(tiff_file_path)

    def show_tiff_file(self, tiff_file_path):
        img_sequence = Image.open(tiff_file_path)
        num_frames = len(img_sequence)

        if self.canvas is not None:
            self.layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

        if self.slider is not None:
            self.layout.removeWidget(self.slider)
            self.slider.deleteLater()

        self.canvas = FigureCanvas(plt.figure())
        self.layout.addWidget(self.canvas)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(num_frames - 1)
        self.slider.valueChanged.connect(self.update_image)
        self.layout.addWidget(self.slider)

        self.update_image(0)

    def update_image(self, page):
        if self.canvas is not None:
            img_sequence.seek(page)
            img_data = img_sequence.convert("RGBA")
            plt.clf()
            plt.imshow(img_data)
            self.canvas.draw()
            self.current_page = page

def main():
    app = QApplication(sys.argv)
    viewer = TIFFViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
