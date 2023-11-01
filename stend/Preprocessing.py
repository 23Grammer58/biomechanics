from tqdm import tqdm
import cv2 as cv
import timeit
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
import os
import json
import glob
import re
import shutil
import csv


class NumpyEncoder(json.JSONEncoder):
    '''
    Класс для  записи в json, его лучше не трогать
    '''

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Calib:
    def __init__(self):
        self.path_calib_params = None
        self. calibration_param = None

    def check_path_calib_params(self):
        print('Укажите путь до файла')
        self.path_calib_params = input()
        if self.path_calib_params.split('.')[-1].strip() != "json":
            print('Yказан неверный формат файла \nВвести еще раз путь?')
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

    def read_param_from_json(self):
        if self.calibration_param is not None:
            return self.calibration_param
        path = self.check_path_calib_params()
        if path is not None:
            self.calibration_param = Calib.read_json(path)
            if type(self.calibration_param) == dict:
                # print('main if',type(self.calibration_param))
                return self.calibration_param
            elif self.calibration_param is None:
                self.read_param_from_json()
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

        size_of_chessboard_squares_mm = 1.5
        objp = objp * size_of_chessboard_squares_mm

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob(os.path.join('stend',  'Calibration4', '*.jpg'))


        for image in tqdm(images):
            img = cv.imread(image)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

            # If found, add object points, image points (after refining them)
            if ret:
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
            self.calibration_param = self.read_param_from_json()
            return self.calibration_param
        if answer == '2':
            print('Введите размеры шахматной доски')
            chessboardSize = tuple(map(int, input().split()))
            self.calibration_param = self.calibration_params(name_protocol, path, chessboardSize)
            return self.calibration_param
        else:
            self.main(name_protocol, path)
        return self.calibration_param



class DataProcessing:
    def __init__(self,path:str,name_protocol:str):
        self.path = path
        self.name_protocol = name_protocol
        self.path_log = self.__path_log()
        self.path_log_new =self.__path_log_new()
        self.paths = self.__sort_in_folder()

    def __path_log(self):
        logs = glob.glob(os.path.join(self.path,'*.log'))
        for val in logs:
            if not re.search(f"(new|{self.name_protocol})", os.path.split(val)[1]):
                print(val)
                return val
        # print(logs)
    def __path_log_new(self):
        log_file = list(os.path.split(self.path_log))
        # print(log_file)
        # log_name = log_file[1].split('.')
        # log_name[0]= log_name[0] +'_new'
        # path_log_new = '.'.join(log_name)
        # new_path_log = os.path.join(log_file[0],path_log_new)
        log_file[1] = re.sub(r'(.+).(log)', r'\1_new.\2', log_file[1])
        new_path_log = os.path.join(log_file[0],log_file[1])
        return new_path_log


    @staticmethod
    def sort_by(file:str) -> float or int:
        '''
        функция для сортировки при считывания файлов
        '''
        # return int(re.search(r'\d{3,}', file)[0])
        # os.path.split(self.path)
        # return int(file.split('/')[-1].split('.')[0])
        # os.path.split(file)[1].split('.')[0])
        file = os.path.split(file)[1]
        name_file = re.sub(r'([\d+\.]?\d+).json|([\d+\.]?\d+).npy', r'\1\2', file)
        # print(name_file)
        return float(name_file)

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
            files = glob.glob(os.path.join(self.path, f"*.{extension}"))
            print(f"[*] Найдено {len(files)} Файлов с раширением {extension}.")
            if not os.path.isdir(os.path.join(self.path, folder_image)):
                os.mkdir(os.path.join(self.path, folder_image))
                print(f"[+] Создана папка {folder_image}.")
            folder_location = os.path.join(self.path, folder_image)
            paths.append(folder_location)

            for file in files:
                # if 'exp_params'in file:
                #     continue
                if re.fullmatch(f'.*\d+\.?\d+?.{extension}|(?!.*exp_params.json)', file) is not None:
                    nowlocation = os.path.basename(file)
                    dst = os.path.join(self.path, folder_image, nowlocation)
                    print(f"[*] Перенесен файл '{file}' в {dst}")
                    # print(dst)
                    shutil.move(file, dst)
        return paths

    def read_tenzo_value_log_to_dictionary(self):
        self.sampling = []
        self.tenzo_data = []
        sum_sampling = 0
        file_lines = 0
        with open(f'{self.path_log}') as datalog:
            iter = 1
            self.tenzo_data.append({0: [], 1: [], 2: [], 3: []})
            for index, value in enumerate(datalog):
                data = value.split(',')
                for i,val in enumerate (data[-4:]):
                    res = re.findall(r'-?\d+', val)
                    if int(data[1]) == iter:
                        self.tenzo_data[iter-1][i].append(float(*res))
                    else:
                        #блок для вычисления дискретизации
                        if len(self.sampling) == 0:
                            self.sampling.append(index)
                        else:
                            sum_sampling += self.sampling[-1]
                            self.sampling.append(index - sum_sampling)
                        iter += 1

                        self.tenzo_data.append({0: [], 1: [], 2: [], 3: []})
                        if int(data[1]) == iter:
                            self.tenzo_data[iter-1][i].append(float(*res))
                file_lines = index
            datalog.close()
        sum_sampling += self.sampling[-1]
        self.sampling.append(file_lines - sum_sampling)

    @staticmethod
    def find_outliers_iqr(data):
        sorted_data = sorted(data)
        q1, q3 = np.percentile(sorted_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        filtered_data = [x for x in data if (lower_bound < x < upper_bound)]
        return filtered_data
        # for i, x in enumerate(data):
        #     #     print(f'{data=}\n,{lower_bound=},{upper_bound=}')
        #     if x < lower_bound or x > upper_bound:
        #         # print(x)
        #         data.pop(i)
        # if abs(max(data)) >150000:
        #     print('posle', data)
        # return data

    @staticmethod
    def find_outliers_z_score(data):
        mean = np.mean(data)
        std = np.std(data)
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        filtered_data = [x for x in data if (lower_bound < x < upper_bound)]
        return filtered_data

    def search_outlier(self):
        for index, window in enumerate(self.tenzo_data):
            for key in window.keys():
                values = np.array(window[key]) * (-1)
                if len(values) > 3:
                    # values = DataProcessing.find_outliers_z_score(list(set(values)))
                    values = DataProcessing.find_outliers_iqr(list(set(values)))
                    # print(f'{values=},\t std = {np.std(values)},\t len = {len(values)}\n')
                window[key] = values

    def mean_value_tenzo(self):
        path_Indication = self.paths[1]
        self.json_files=glob.glob(os.path.join(path_Indication,'*.json'))
        # coef = [46745, 45731, 45731, 39700]
        self.json_files.sort(key = DataProcessing.sort_by)
        print(self.json_files)
        self.json_files_average_strength = {}
        # print(self.tenzo_data)
        self.error_open = {}
        for index, window in enumerate(self.tenzo_data):
            with open(f'{self.json_files[index]}' ) as json_file:

                try:
                    file = eval(json.load(json_file))
                    file[f'{index + 1}']
                except Exception as e:
                    # print(self.json_files[index])
                    # print(file)
                    # print(index+1)
                    self.error_open[index+1] = e
                    continue
                tenzo_mean_value = []
                for key in window.keys():
                    values = np.array(window[key])
                    # print(f'{window=},{values=},{key=}')
                    mean = sum(values)/len(values)
                    tenzo_mean_value.append(mean)
                # for i in range (len(tenzo_mean_value)):
                #     tenzo_mean_value[i]  = tenzo_mean_value[i]/coef[i]
                # self.file[index+1] = [self.move[index], tenzo_mean_value]
                file[f'{index + 1}'][1] = tenzo_mean_value
            self.json_files_average_strength.update(file)
        # print(self.json_files_average_strength)
                # print(f'{tenzo_mean_value=}')

    def rewriting_file_json(self):
        # print(self.error_open)
        for index, file in enumerate(self.json_files):
            with open(f'{file}','w') as json_file:
                if index+1 in self.error_open:
                    d = {index+1: f'Reading error  {self.error_open[index+1]}'}
                    ff = json.dump(f'{d}',json_file)
                    print(ff)
                    break
                    # print(json.dumps(f"{index + 1}:'Reading error  {self.error_open[index + 1]}'"))

                else:
                    values=self.json_files_average_strength[f'{index + 1}']
                    d = {f'{index + 1}':values}
                    json.dump(f'{d}',json_file)
                    # print(json.dumps(f"{index + 1}:{values}",indent=4))

    def rewriting_file_log(self):
        with open(f'{self.path_log}') as datalog:
            with open(f'{self.path_log_new}','w', encoding='utf-8') as new_datalog:
                iter = 1
                for index, value in enumerate(datalog):
                    data = value.split(',')
                    move_1_3 = np.array([])
                    move_2_4 =  np.array([])
                    for i, val in enumerate(data[2:6]):
                        res = re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)', val)
                        if i == 0 or i == 1:
                            move_1_3 = np.append(move_1_3,float(*res))
                        elif i == 2 or i == 3:
                            move_2_4 = np.append(move_2_4,float(*res))
                    if int(data[1]) == iter:
                        move_1_3 = move_1_3 / self.sampling[iter-1]
                        move_2_4 = move_2_4 / self.sampling[iter-1]
                    else:
                        iter += 1
                        if int(data[1]) == iter:
                            move_1_3 = move_1_3 / self.sampling[iter - 1]
                            move_2_4 = move_2_4 / self.sampling[iter - 1]
                        else:
                            print('dddddddd')
                    data_str_datalog = [tuple(move_1_3),tuple(move_2_4)]
                    res = re.sub(r'\(\([-+]?\d+\.\d+, [-+]?\d+\.\d+\), \([-+]?\d+\.\d+, [-+]?\d+\.\d+\)\)', f'({data_str_datalog[0]}, {data_str_datalog[1]})',value)
                    new_datalog.write(res)

            datalog.close()
            new_datalog.close()


    def create_csv(self):
        data = {}
        names = ['frame', 'axis_0_2', 'axis_1_3', 'tenzo_0', 'tenzo_1', 'tenzo_2', 'tenzo_3', 'mean_tenzo_0_2',
                 'mean_tenzo_1_3']

        for name in names:
                data[name] = []

        for id, x in enumerate(self.json_files_average_strength):
            data['frame'].append(x)
            value_key_x = self.json_files_average_strength[x]
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
            data['tenzo_1'].append(value_tenzo[1])
            data['tenzo_2'].append(value_tenzo[2])
            data['tenzo_0'].append(value_tenzo[0])
            data['tenzo_3'].append(value_tenzo[3])
            data['mean_tenzo_0_2'].append((value_tenzo[0] + value_tenzo[2]) / 2)
            data['mean_tenzo_1_3'].append((value_tenzo[1] + value_tenzo[3]) / 2)

            if id == 0:
                data['axis_0_2'].append(sum_0_2)
                data['axis_1_3'].append(sum_1_3)

            else:
                data['axis_0_2'].append(round(data['axis_0_2'][id - 1] + sum_0_2, 2))
                data['axis_1_3'].append(round(data['axis_1_3'][id - 1] + sum_1_3, 2))

        data['correlation_force_0_2'] = np.array(data['mean_tenzo_0_2']) * 0.96713
        data['correlation_force_1_3'] = np.array(data['mean_tenzo_1_3']) * 0.88

        with open(fr'{os.path.join(self.path, self.name_protocol)}.csv', 'w', newline='') as csvfile:
            fieldnames = list(data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(data['frame'])):
                row = {}
                for key in data:
                    # print(key)
                    row[key] = data[key][i]
                # print(row)
                writer.writerow(row)
        self.data = data
        print(fr'создан csv файл {os.path.join(self.path, self.name_protocol)}.csv')
    def draw_plot(self):
        x_0_2 = self.data['axis_0_2']
        y_0_2 = self.data['mean_tenzo_0_2']

        fig, ax = plt.subplots(figsize=(150, 30), dpi=150)

        plt.plot(x_0_2, y_0_2, marker='x', label = 'axis 0-2')
        plt.ylabel('$F($N$)$', rotation = 0)
        plt.xlabel('Displacement [mm]')
        plt.title(f'{self.name_protocol}')
        # plt.scatter(x, y)
        # plt.xticks(x[::10], rotation=45)
        # plt.stem(x,y)
        # for row in tenzo_3:
        # for id, value in enumerate(y):
        #     # print(row.cty)
        #     ax.text(id, value, s=round(value, 2), horizontalalignment='center', verticalalignment='bottom', fontsize=8)
        # self.axis_labels(x, y)
        x_1_3 = self.data['axis_1_3']
        y_1_3 = self.data['mean_tenzo_1_3']
        plt.plot(x_1_3, y_1_3, marker='*', label = 'axis 1-3' )
        plt.legend()
        plt.show()


class WorkToImages:
    """
    Класс для работы с изображениями:
    калибровка,сохранение ....
    """
    def __init__(self, path: str, paths: list, calibration_params: dict | None, look_images: list | str):

        if isinstance(calibration_params, dict):
            self.calibration_params = calibration_params
        else:
            # print("Нет калибровочных параметров")
                    # "Опреации с изображенями будут произведены без учета калибровочных параметров")
            self.calibration_params = None
        self.path = path
        self.paths = paths
        self.calibration_params = calibration_params
        self.path_folder_frame = self.paths[0]
        self.frames_paths = self.__npy_files()
        self.__get_index(look_images)

    def __npy_files(self):
        frames_path = glob.glob(f'{self.path_folder_frame}/*.npy')  # path to npy
        frames_path.sort(key=DataProcessing.sort_by)
        return frames_path

    def __get_index(self, look_images):
        if isinstance(look_images,str):
            if look_images == 'all':
                self.look_images = list(range(1, len(self.frames_paths)+1))
            if look_images == 'no':
                self.look_images = None
        elif isinstance(look_images, list):
            if len(look_images) > 0:
                self.look_images = look_images
            else:
                self.look_images = None

    def look(self, index: int):
        '''
        функция  для просмтора изображений если двруг захотим
        '''
        print(self.frames_paths)
        npy = np.load(self.frames_paths[index-1])  # read npy
        # npy = cv2.cvtColor(npy, cv2.COLOR_BGR2RGB)
        print(npy)
        img = Image.fromarray(npy)
        img.save(f"{self.paths[2]}/{index}.tif")  # transfer from npy to tif

    def calibration_image(self, frame : str, index: int, look_image=False):
        '''
        калибровка изображения по калибровочным пармаетрам  и возможно сохранение при желании
        '''

        folder_images = 'Calibration_Images'
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

    def conversion_to_grayscale(self, frame, index: int, look_image=False, flag_calib=True):
        '''
        перевод изображения в градации серого  и сохранепия при желании
        '''
        if flag_calib:
            folder_images = 'Grayscale_Calib_Images'
        else:
            folder_images = 'Grayscale_No_Calib_Images'
        if look_image:
            if not os.path.isdir(os.path.join(self.path, folder_images)):
                os.mkdir(os.path.join(self.path, folder_images))
            folder_location_image = os.path.join(self.path, folder_images)

        frame = np.average(frame, axis = 2, weights=[0.144, 0.587, 0.299])
        frame = frame.astype(np.uint8)
        # print(self.paths)

        np.save(f'{self.paths[3]}/{index}', frame)

        if look_image:
            frame = frame.astype(np.uint8)
            frame = Image.fromarray(frame)
            frame.save(f'{folder_location_image}/{index}.tif')

    def processing_answer_calib(self):
        print("Перевести в градации серого неоткалиброванные изображения?\n"
              " 1 - Продолжить, 0 - Завершить.")
        answer = input()
        if answer.strip().casefold() in ['да', 'yes', '1']:
            return True
        elif answer.strip().casefold() in ['нет', 'no', '0']:
            return False
        else:
            return self.processing_answer_calib()
    def main_function(self):
        '''
        тут калибровка и перевод в градации серого, беря каждый элемент из папки в которую мы отсортировали выше и удаление первоначальных фреймов для оптимизации по памяти
        '''
        # print('Продолжить работу с изображенями без учета калибровочных параметров?')
        flag_calib = True

        if self.calibration_params is None:
            print("Нет калибровочных параметров")
            answer = self.processing_answer_calib()
            if answer:
                flag_calib = False
            elif not answer:
                return
        len_files = len(self.frames_paths)
        print(f'Колличество файлов:{len_files}')
        for index, frame in enumerate(self.frames_paths):
            print(f'frame:{frame}, index:{index+1}, Осталось обработать файлов: {len_files - (index )}')
            if flag_calib:
                npy_file = self.calibration_image(frame, index + 1)
            else:
                npy_file = np.load(frame)
            if self.look_images is not None:
                if (index+1) in self.look_images:
                    look_image = True
                    self.conversion_to_grayscale(npy_file, index+1,  look_image, flag_calib)
                else:
                    look_image = False
                    self.conversion_to_grayscale(npy_file, index+1, look_image, flag_calib)
            else:
                look_image = False
                self.conversion_to_grayscale(npy_file, index + 1, look_image, flag_calib)

            # os.remove(f'{frame}')
        # os.rmdir(self.paths[0])


def test():
    path = 'syfooq'
    name_protocol = 'syfooq'
    calib = Calib()
    start_main = timeit.default_timer()
    calibration_param = calib.main(name_protocol, path)

    tt = DataProcessing(path,name_protocol)
    # tt.read_tenzo_value_log_to_dictionary()
    # tt.search_outlier()
    # tt.mean_value_tenzo()
    # tt.rewriting_file_json()
    # tt.rewriting_file_log()
    # tt.create_csv()
    # tt.draw_plot()

    #___________
    gg = WorkToImages(path, tt.paths, calibration_param,[1,10])
    gg.look(100)
    # gg.main_function()

if __name__ == "__main__":
    test()