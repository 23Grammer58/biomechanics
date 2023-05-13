import glob
import json
import os
import shutil
import timeit
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import logging
import cv2 as cv
import csv


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def calibration_params(name_protocol,path):
    start = timeit.default_timer()
    calib_params = {}
    chessboardSize = (15,8)
    frameSize = (4096,3000)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 5
    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('CalibrationOpenCV/*.jpg')
    # print(images)

    for image in images:
        # print(image)
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            #cv.imshow('img', img)
            #cv.waitKey(1000)

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
    calib_params['Distorsion_Parameters']= dist
    calib_params['Rotation_Vectors'] = rvecs
    calib_params['Translation_Vectors'] = tvecs
    json_calib_params = json.dumps(calib_params, indent = 4, cls=NumpyEncoder)
    with open(f'{os.path.join(path,name_protocol)}.json', 'w') as outfile:
        outfile.write(json_calib_params)
        # json.dump(json_calib_params, outfile)
    end = timeit.default_timer()
    print(f'Parameter calibration time:{end - start}')
    return calib_params


class Sort:
    def __init__(self,path:str,name_protocol:str, calib_params: dict,look_image_list = []):
        self.path = path
        self.name_protocol = name_protocol
        self.look_images_list = look_image_list
        self.calibration_params = calib_params
        self.paths = self.__sort_in_folder()
        self.data = {}

    def __sort_by(self,file: str) -> int:
        return int(re.search(r'\d{3,}', file)[0])
    def __sort_in_folder(self):
        paths = []  # add paths all to folders

        extensions = {
            "npy": "Frame",
            "json": "Indication",
            "tif": "Images",
            " ": "Grayscale"
        }

        for extension, folder_image in extensions.items():
            files = glob.glob(os.path.join(self.path, fr"*^{self.name_protocol}.{extension}"))
            print(f"[*] Найдено {len(files)} Файлов с раширением {extension}.")

            if not os.path.isdir(os.path.join(self.path, folder_image)):
                # print(folder_image)
                os.mkdir(os.path.join(self.path, folder_image))
                print(f"[+] Создана папка {folder_image}.")
            folder_location = os.path.join(self.path, folder_image)


            for file in files:
                nowlocation = os.path.basename(file)
                dst = os.path.join(self.path, folder_image, nowlocation)
                print(f"[*] Перенесен файл '{file}' в {dst}")
                # print(dst)
                shutil.move(file, dst)
            paths.append(folder_location)
        return paths

    def write_json_to_dictionary(self):
        logging.basicConfig(level=logging.INFO,filename=f"{os.path.join(path,name_protocol)}.log", filemode="w",
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

        frame_x = []
        axis_0_2 = []
        axis_1_3 = []
        tenzo_0 = []
        tenzo_1 = []
        tenzo_2 = []
        tenzo_3 = []
        mean_tenzo_0_2 = []
        mean_tenzo_1_3 = []
        for id, x in enumerate(all_json):
            frame_x.append(x)
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
            # print(value_tenzo)
            tenzo_0.append(abs(value_tenzo[0]))
            tenzo_1.append(abs(value_tenzo[1]))
            tenzo_2.append(abs(value_tenzo[2]))
            tenzo_3.append(abs(value_tenzo[3]))
            mean_tenzo_0_2.append((abs(value_tenzo[0]) + abs(value_tenzo[2])) / 2)
            mean_tenzo_1_3.append((abs(value_tenzo[1]) + abs(value_tenzo[3])) / 2)

            if id == 0:
                axis_0_2.append(sum_0_2)
                axis_1_3.append(sum_1_3)

            else:
                axis_0_2.append(round(axis_0_2[id - 1] + sum_0_2, 2))
                axis_1_3.append(round(axis_1_3[id - 1] + sum_1_3, 2))
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
        for i in range(len(mean_tenzo_0_2)):
            if mean_tenzo_0_2[i] > 0.8:
                mean_tenzo_0_2[i] = (mean_tenzo_0_2[i - 1] + mean_tenzo_0_2[i + 1]) / 2
            if mean_tenzo_1_3[i] > 0.8:
                mean_tenzo_1_3[i] = (mean_tenzo_1_3[i - 1] + mean_tenzo_1_3[i + 1]) / 2

        self.data['frame_x'] = frame_x
        self.data['axis_0_2'] = axis_0_2
        self.data['axis_1_3'] = axis_1_3
        self.data['tenzo_0'] = tenzo_0
        self.data['tenzo_1'] = tenzo_1
        self.data['tenzo_2'] = tenzo_2
        self.data['tenzo_3'] = tenzo_3
        self.data['Force_0_2'] = mean_tenzo_0_2
        self.data['Force_1_3'] = mean_tenzo_1_3
        self.data['correlation_force_0_2'] = np.array(mean_tenzo_0_2) * 0.96713
        self.data['correlation_force_1_3'] = np.array(mean_tenzo_1_3) * 0.88

        with open(fr'{os.path.join(path,name_protocol)}.csv', 'w', newline='') as csvfile:
            fieldnames = list(self.data.keys())
            writer = csv.DictWriter(csvfile,fieldnames = fieldnames)
            writer.writeheader()
            for i in range(len(self.data['frame_x'])):
                row = {}
                for key in self.data:
                    # print(key)
                    row[key] = self.data[key][i]
                # print(row)
                writer.writerow(row)


    def look(self,frame, index:int):
        npy = np.load(frame)  # read npy
        # npy = cv2.cvtColor(npy, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(npy)
        img.save(f"{self.paths[2]}/{index}.tif")  # transfer from npy to tif

    def calibration_image(self,frame, index:int,look_image = False ):
        folder_images = 'Calibration_images'
        if look_image:
            if not os.path.isdir(os.path.join(path, folder_images)):
                os.mkdir(os.path.join(path, folder_images))
            folder_location_image = os.path.join(path, folder_images)
        start = timeit.default_timer()

        # start = timeit.default_timer()
        # img = cv.imread(frame)
        img = np.load(frame)
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(self.calibration_params.get('Camera_Matrix'), self.calibration_params.get('Distorsion_Parameters'), (w, h), 1, (w, h))
        # print("roi",roi)
        # print("\nNewCamera Matrix:\n", newCameraMatrix)

        # Undistort
        dst = cv.undistort(img, self.calibration_params.get('Camera_Matrix'), self.calibration_params.get('Distorsion_Parameters'), None, newCameraMatrix)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        # cv.imwrite('caliResult3.jpg', dst)

        # Undistort with Remapping
        mapx, mapy = cv.initUndistortRectifyMap(self.calibration_params.get('Camera_Matrix'), self.calibration_params.get('Distortion_Parameters'), None, newCameraMatrix, (w, h), 5)
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
        mean_error = 0

        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error

        print( "total error: {}".format(mean_error/len(objpoints)) )

    def conversion_to_grayscale(self,frame, index:int, look_image = False):
        # folder_npy = 'grayscale_npy'
        folder_images = 'grayscale_image'
        if look_image:
            if not os.path.isdir(os.path.join(path, folder_images)):
                os.mkdir(os.path.join(path, folder_images))
            folder_location_image = os.path.join(path, folder_images)


        frame = np.average(frame, axis=2, weights = [0.144, 0.587, 0.299])
        frame = frame.astype(np.uint8)
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

    def draw_plot(self,):
        x = self.data.get('axis_0_2')[:200]
        y = self.data.get('Force_0_2')[:200]
        # print('x',len(x))
        # print('y',y)
        fig, ax = plt.subplots(figsize=(150, 30), dpi=150)
        plt.ylim(0,0.3)
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

    def main_function(self,):
        frames_path = glob.glob(f'{self.paths[0]}/*.npy')  # path to npy
        frames_path.sort(key=self.__sort_by)
        index = 1
        for frame in frames_path:
            print(f'frame:{frame}, index:{index}')
            # img = np.load(frame)
            # print()
            look_image = False
            if index in self.look_images_list:
                look_image = True
                self.look(frame,index)
            calib_npy = self.calibration_image(frame,index,look_image)
            self.conversion_to_grayscale(calib_npy,index,look_image)
            os.remove(f'{frame}')
            index += 1
        os.rmdir(self.paths[0])


if __name__ == "__main__":
    path = '/home/ali/Desktop/test'
    name_protocol = 'test'
    look_images = []
    start_main = timeit.default_timer()
    if len(look_images) > 0:
        t = Sort(path,name_protocol,calibration_params(name_protocol,path),look_images)
    else:
        t = Sort(path, name_protocol, calibration_params(name_protocol,path))
    t.write_json_to_dictionary()
    # t.main_function()
    t.draw_plot()
    end_main = timeit.default_timer()
    print(f'Program running time:{(end_main - start_main)//60} : {(end_main - start_main) % 60}')