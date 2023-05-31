import csv
import glob
import json
import logging
import os
import re
import shutil
import timeit
from typing import List
from pathlib import Path
import datetime
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

class NumpyEncoder(json.JSONEncoder):
    '''
    Класс для  записи в json, его лучше не трогать
    '''

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Sort:
    def __init__(self, path: str, name_protocol: str, calib_params: dict | None, look_image_list: List = []):
        self.path = path
        self.name_protocol = name_protocol
        self.look_images_list = look_image_list
        self.calibration_params = calib_params
        self.paths = self.__sort_in_folder()
        self.data = {}

    def __sort_by(self, file: str) -> int:
        '''
        функция для сортировки при считывания файлов
        '''
        return int(re.search(r'\d{3,}', file)[0])

    def __sort_in_folder(self):

        '''
        функция для создания папочек и сортировки файлов по ним
        '''

        paths = []  # add paths all to folders

        extensions = {
            "npy": "Frame",
            "json": "Indication",
            "tif": "Images",
            " ": "Grayscale"
        }

        for extension, folder_image in extensions.items():
            files = glob.glob(os.path.join(self.path, fr"*.{extension}"))
            print(f"[*] Найдено {len(files)} Файлов с раширением {extension}.")
            if not os.path.isdir(os.path.join(self.path, folder_image)):
                # print(folder_image)
                os.mkdir(os.path.join(self.path, folder_image))
                print(f"[+] Создана папка {folder_image}.")
            folder_location = os.path.join(self.path, folder_image)
            paths.append(folder_location)

            for file in files:
                print(file)
                if re.fullmatch(f'.*\d*\.\d*\.{extension}|.*exp_params.json', file) is not None:
                    nowlocation = os.path.basename(file)
                    dst = os.path.join(self.path, folder_image, nowlocation)
                    print(f"[*] Перенесен файл '{file}' в {dst}")
                    # print(dst)
                    shutil.move(file, dst)
        return paths

    def write_json_to_dictionary(self):
        '''
        парсинг json
        считывание ошибок и запись в логгер
        запись в csv
        '''
        logging.basicConfig(level=logging.INFO, filename=f"{os.path.join(self.path, self.name_protocol)}.log", filemode="w",
                            format="%(asctime)s %(levelname)s %(message)s")
        logging.info(f'Name protocol \'{self.name_protocol}\'')
        # print(self.paths)
        path_Indication = self.paths[1]
        list_json = sorted(os.listdir(path_Indication))
        all_json = {}
        id = 1
        for x in list_json:
            path_element_Inication = os.path.join(path_Indication, x)
            if "exp_params" not in path_element_Inication:
                with open(path_element_Inication, 'r') as f:
                    try:
                        dictionary_string = eval(json.load(f))
                    except SyntaxError as err:
                        logging.error(f"writing error frame {id}")
                        id += 1
                        continue
                    try:
                        value_key = dictionary_string[fr'{id}']
                    except KeyError as err:
                        logging.error(f"frame skipped {id}")
                        id += 1
                        continue
                    all_json[f"{id}"] = value_key
                    id += 1
        # return all_json

        names = ['frame', 'axis_0_2', 'axis_1_3', 'tenzo_0', 'tenzo_1', 'tenzo_2', 'tenzo_3', 'mean_tenzo_0_2',
                 'mean_tenzo_1_3']
        for name in names:
            self.data[name] = []

        for id, x in enumerate(all_json):
            self.data['frame'].append(x)
            value_key_x = all_json[x]
            value_axes = value_key_x[0]
            value_tenzo = value_key_x[1]
            value_axis_0_2 = value_axes[0]
            value_axis_1_3 = value_axes[1]
            value_axis_0 = value_axis_0_2[0]
            value_axis_2 = value_axis_0_2[1]
            value_axis_1 = value_axis_1_3[0]
            value_axis_3 = value_axis_1_3[1]
            sum_0_2 = round(value_axis_0 + value_axis_2, 2)
            sum_1_3 = round(value_axis_1 + value_axis_3, 2)
            self.data['tenzo_0'].append(abs(value_tenzo[0]))
            self.data['tenzo_1'].append(abs(value_tenzo[1]))
            self.data['tenzo_2'].append(abs(value_tenzo[2]))
            self.data['tenzo_3'].append(abs(value_tenzo[3]))
            self.data['mean_tenzo_0_2'].append((abs(value_tenzo[0]) + abs(value_tenzo[2])) / 2)
            self.data['mean_tenzo_1_3'].append((abs(value_tenzo[1]) + abs(value_tenzo[3])) / 2)

            if id == 0:
                self.data['axis_0_2'].append(sum_0_2)
                self.data['axis_1_3'].append(sum_1_3)

            else:
                self.data['axis_0_2'].append(round(self.data['axis_0_2'][id - 1] + sum_0_2, 2))
                self.data['axis_1_3'].append(round(self.data['axis_1_3'][id - 1] + sum_1_3, 2))
        # print('0_2',axis_0_2)
        # print('1_3',axis_1_3)
        # print('x',frame_x)
        # print('tenzo 0',tenzo_0)
        # print('tenzo 1',tenzo_1)
        # print('tenzo 2',tenzo_2)
        # print('tenzo 3',tenzo_3)

        # print("0:",value_axis_0)
        # print("1:",value_axis_1)
        # print("2:",value_axis_2)
        # print("3:",value_axis_3)
        # -----------------------'полнешйий костыль для калибровки'------------------------------------
        # for i in range(len(mean_tenzo_0_2)):
        #     if mean_tenzo_0_2[i] > 0.8:
        #         mean_tenzo_0_2[i] = (mean_tenzo_0_2[i - 1] + mean_tenzo_0_2[i + 1]) / 2
        #     if mean_tenzo_1_3[i] > 0.8:
        #         mean_tenzo_1_3[i] = (mean_tenzo_1_3[i - 1] + mean_tenzo_1_3[i + 1]) / 2
        # ---------------------------------------------------------------------------

        self.data['correlation_force_0_2'] = np.array(self.data['mean_tenzo_0_2']) * 0.96713
        self.data['correlation_force_1_3'] = np.array(self.data['mean_tenzo_1_3']) * 0.88

        with open(fr'{os.path.join(self.path, self.name_protocol)}.csv', 'w', newline='') as csvfile:
            fieldnames = list(self.data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(self.data['frame'])):
                row = {}
                for key in self.data:
                    # print(key)
                    row[key] = self.data[key][i]
                # print(row)
                writer.writerow(row)
        print(fr'создан csv файл {os.path.join(self.path, self.name_protocol)}.csv')
        print(fr'создан logger {os.path.join(self.path, self.name_protocol)}.log')

    def look(self, frame, index: int):
        '''
        функция  для просмтора изображений если двруг захотим
        '''
        npy = np.load(frame)  # read npy
        # npy = cv2.cvtColor(npy, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(npy)
        img.save(f"{self.paths[2]}/{index}.tif")  # transfer from npy to tif

    def calibration_image(self, frame, index: int, look_image=False):
        '''
        калибровка изображения по калибровочным пармаетрам  и возможно сохранение при желании
        '''

        folder_images = 'Calibration_images'
        if look_image:
            if not os.path.isdir(os.path.join(self.path, folder_images)):
                os.mkdir(os.path.join(self.path, folder_images))
            folder_location_image = os.path.join(self.path, folder_images)
        start = timeit.default_timer()

        # start = timeit.default_timer()
        # img = cv.imread(frame)
        img = np.load(frame)
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(self.calibration_params.get('Camera_Matrix'),
                                                            self.calibration_params.get('Distorsion_Parameters'),
                                                            (w, h), 1, (w, h))
        # print("roi",roi)
        # print("\nNewCamera Matrix:\n", newCameraMatrix)

        # Undistort
        dst = cv.undistort(img, self.calibration_params.get('Camera_Matrix'),
                           self.calibration_params.get('Distorsion_Parameters'), None, newCameraMatrix)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        # cv.imwrite('caliResult3.jpg', dst)

        # Undistort with Remapping
        mapx, mapy = cv.initUndistortRectifyMap(self.calibration_params.get('Camera_Matrix'),
                                                self.calibration_params.get('Distortion_Parameters'), None,
                                                newCameraMatrix, (w, h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        # textlookfor = r'[0-9]+'
        # res = re.search(textlookfor, frame[-7:-3])
        # print(res[0])

        # np.save(f'{folder_location_npy}/frame{j}', dst)
        if look_image:
            cv.imwrite(f'{folder_location_image}/frame{index}.tif', dst)

        end = timeit.default_timer()
        print(f"Time taken is colibration image {end - start}s")
        return dst
        # Reprojection Error
        # mean_error = 0
        #
        # for i in range(len(objpoints)):
        #     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        #     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        #     mean_error += error
        #
        # print("total error: {}".format(mean_error / len(objpoints)))

    def conversion_to_grayscale(self, frame, index: int, look_image=False):
        '''
        перевод изображения в градации серого  и сохранепия при желании
        '''
        folder_images = 'grayscale_image'
        if look_image:
            if not os.path.isdir(os.path.join(self.path, folder_images)):
                os.mkdir(os.path.join(self.path, folder_images))
            folder_location_image = os.path.join(self.path, folder_images)

        frame = np.average(frame, axis=2, weights=[0.144, 0.587, 0.299])
        frame = frame.astype(np.uint8)
        print(self.paths)

        np.save(f'{self.paths[3]}/{index}', frame)

        if look_image:
            frame = frame.astype(np.uint8)
            frame = Image.fromarray(frame)
            frame.save(f'{folder_location_image}/{index}.tif')

    # def axis_labels(self, x, y):
    #     if x == self.data[0]:
    #         plt.xlabel('frame', fontsize=14)
    #     if y == self.data[1]:
    #         plt.ylabel('stretching by sensor 0 2', fontsize=14)
    #     if y == self.data[2]:
    #         plt.ylabel('stretching by sensor 1 3', fontsize=14)
    #     if y == self.data[3]:
    #         plt.ylabel('force reading of sensor 0', fontsize=14)
    #     if y == self.data[4]:
    #         plt.ylabel('force reading of sensor 1', fontsize=14)
    #     if y == self.data[5]:
    #         plt.ylabel('force reading of sensor 2', fontsize=14)
    #     if y == self.data[6]:
    #         plt.ylabel('force reading of sensor 3', fontsize=14)
    #

    def draw_plot(self, ):
        '''
        тут можно порисовать всякие графики, позже  поприкольнее сделаю
        '''
        x = self.data.get('axis_0_2')[:200]
        y = self.data.get('Force_0_2')[:200]
        # print('x',len(x))
        # print('y',y)
        fig, ax = plt.subplots(figsize=(150, 30), dpi=150)
        plt.ylim(0, 0.3)
        plt.plot(x, y)
        # plt.scatter(x, y)
        plt.xticks(x[::5], rotation=45)
        # plt.stem(frame_x,tenzo_3)
        # for row in tenzo_3:
        # for id, value in enumerate(y):
        #     # print(row.cty)
        #     ax.text(id, value, s=round(value, 2), horizontalalignment='center', verticalalignment='bottom', fontsize=8)
        # self.axis_labels(x, y)
        plt.show()

    def main_function(self, ):
        '''
        тут калибровка и перевод в градации серого, беря каждый элемент из папки в которую мы отсортировали выше и удаление первоначальных фреймов для оптимизации по памяти
        '''
        if self.calibration_params is None:
            return (print("Нет калибровочных параметров "))
        frames_path = glob.glob(f'{self.paths[0]}/*.npy')  # path to npy
        frames_path.sort(key=self.__sort_by)
        index = 1
        for frame in frames_path:
            print(f'frame:{frame}, index:{index}')
            # img = np.load(frame)
            # print()
            look_image = False
            calib_npy = self.calibration_image(frame, index, look_image)
            if index in self.look_images_list:
                look_image = True
                self.look(frame, index)
            self.conversion_to_grayscale(calib_npy, index, look_image)
            # os.remove(f'{frame}')
            index += 1
        # os.rmdir(self.paths[0])


class Calib:
    def __init__(self):
        self.path_calib_params = None
        self. calibration_param = None

    def check_path_calib_params(self):
        print('Укажите путь до файла')
        self.path_calib_params = input()
        if self.path_calib_params.split('.')[-1].strip() != "json":
            print('указан неверный формат файла \nВвести еще раз путь?')
            answer = input().casefold()
            if answer.strip() in ("yes", "да"):
                self.check_path_calib_params()
            elif answer.strip() in ("no", "нет"):
                self.path_calib_params = None
                return self.path_calib_params
            else:
                self.check_path_calib_params()
        else:
            return self.path_calib_params
        return self.path_calib_params

    @staticmethod
    def read_json(path:Path|str):
        filename = path.split('/')[-1]
        try:
            with open(path, 'r') as f:
                colib_params = json.load(f)
        except FileNotFoundError:
            sys.stderr.write(f'{filename} not found in directory \n')
            return None
        # print(colib_params.keys())
        # print('read',type(colib_params))
        return colib_params

    def main_2(self):
        if self.calibration_param is not None:
            return self.calibration_param
        path = self.check_path_calib_params()
        if path is not None:
            self.calibration_param = Calib.read_json(path)
            if type(self.calibration_param) == dict:
                # print('main if',type(self.calibration_param))
                return self.calibration_param
            elif self.calibration_param is None:
                self.main_2()
    @staticmethod
    def calibration_params(name_protocol: str, path: Path, chessboardSize: tuple):
        '''
        тут происходит калибровка параметров камеры и запись в json по имени протокал
        в json запишем
        :param name_protocol: Name of protocol
        :param path: Path where to save json
        :return:
        '''
        start = timeit.default_timer()
        calib_params = {}
        # chessboardSize = (15,8)
        frameSize = (4096, 3000)

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

        size_of_chessboard_squares_mm = 5
        objp = objp * size_of_chessboard_squares_mm

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob('stend/CalibrationOpenCV/*.jpg')
        # print('ss',images)

        for image in images:
            # print(image)
            img = cv.imread(image)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
                # cv.imshow('img', img)
                # cv.waitKey(1000)

        cv.destroyAllWindows()

        # print(imgpoints)
        ############## CALIBRATION #######################################################

        ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

        # print("Camera Calibrated: ", ret)
        # print("\nCamera Matrix:\n", cameraMatrix)
        # print("\nDistorsion Parameters:\n", dist)
        # print("\nRotation Vectors:\n", rvecs)
        # print("\nTranslation Vectors:\n", tvecs)
        calib_params['Camera_Calibrated'] = ret
        calib_params['Camera_Matrix'] = cameraMatrix
        calib_params['Distorsion_Parameters'] = dist
        calib_params['Rotation_Vectors'] = rvecs
        calib_params['Translation_Vectors'] = tvecs
        json_calib_params = json.dumps(calib_params, indent=4, cls=NumpyEncoder)
        with open(f'{os.path.join(path, name_protocol)}.json', 'w') as outfile:
            outfile.write(json_calib_params)
            # json.dump(json_calib_params, outfile)
            print(fr'json c параметрами калибровки сохранен в {os.path.join(path, f"{name_protocol}.json")}')
        end = timeit.default_timer()
        print(f'Parameter calibration time:{end - start}')
        return calib_params

    def main(self,name_protocol, path):
        if self.calibration_param is not None:
            return self.calibration_param
        print('Выберите  калибровочные параметры \n 0 - None, 1 - Cчитать с файла, 2 - Запустить процесс подсчета\nВведите цифру ')
        answer = input().strip().casefold()
        if answer == '0':
            self.calibration_param = None
            return self.calibration_param
        if answer == '1':
            self.calibration_param = self.main_2()
            return self.calibration_param
        if answer == '2':
            print('Введите размеры шахматной доски')
            chessboardSize = tuple(map(int, input().split()))
            self.calibration_param = self.calibration_params(name_protocol, path, chessboardSize)
            return self.calibration_param
        else:
            self.main(name_protocol, path)
        return self.calibration_param


if __name__ == "__main__":

    '''
    Главная функция 
    тут весь движ 
    '''

    # -----------------------------Тут вводные параметры протокола-----------------------------------------------------------------
    #                                    |

    path = '/home/ali/Desktop/test'  # задаем путь в нашу папочку с данными
    name_protocol = 'test'  # задаем имя нашего протокола
    look_images = []  # можем записать номера изображений которые хотим глянуть


    #                                    |
    #-----------------------------Тут вводные параметры протокола-----------------------------------------------------------------



    calib  = Calib()
    start_main = timeit.default_timer()
    calibration_param = calib.main(name_protocol,path)

    if len(look_images) > 0:
        t = Sort(path, name_protocol, calibration_param, look_images)
    else:
        t = Sort(path, name_protocol, calibration_param)
    t.write_json_to_dictionary()  # чтения json и запись в csv, создается также логгер с ошибками(неправильная запись,либо пропуск фремйа)
    t.main_function()  # калибрвка изображения и перевод в градации серого а также сохранение  изображений в случае если у нас массив  look_images содежит индексы фреймов
    # t.draw_plot()# отрисовка, параметры нужно через sefl указать в классе, чуть позже напишу  нормально
    end_main = timeit.default_timer()  # замер времени
    print(f'Program running time: {int((end_main - start_main)//60)}:{((end_main - start_main) % 60):0.2f}')

# /media/ali/E616EDC516ED9739/BioMed/biomechanics/stend/hui.json
# /home/ali/Desktop/test/test.json
