import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
paths = glob.glob(r'C:\Users\Lenovo\PycharmProjects\MainBioStand\BioStand\stat\*.csv')
print(paths)
print('-----')
# for i in paths:
#     df = pd.read_csv(i)
#     x = df.frame
#     y_ = df.axis_0_2
#     y1_ = df.axis_1_3
#     y = pd.concat([pd.Series([0]), y_[:-1]], ignore_index=True)
#     y1 = pd.concat([pd.Series([0]), y1_[:-1]], ignore_index=True)
#     fig, ax = plt.subplots(figsize=(150, 30), dpi=300)
#
#     plt.rcParams.update({'font.size': 5})
#
#     plt.plot(x, y, marker='x', label='$X$')
#     plt.ylabel('Displacement(mm)')
#     plt.xlabel('Time(s)')
#
#     plt.title('Load Protocol')
#     plt.plot(x, y1, marker='*', label='$Y$')
#
#     plt.ylabel('Перемещение[мм]')
#     plt.xlabel('Время[с]')
#     plt.legend(title=f'{os.path.split(i)[1].split(".")[0]}')
#     plt.show()
# #              #------------------------------------------------------------------
# for i in paths:
#     df=pd.read_csv(i)
#     y = df['correct_force_X(N)']
#     y1 = df['correct_force_Y(N)']
#     x_ = df.axis_0_2
#     x1_ = df.axis_1_3
#     x = pd.concat([pd.Series([0]), x_[:-1]], ignore_index=True)
#     x1 = pd.concat([pd.Series([0]), x1_[:-1]], ignore_index=True)
#     fig, ax = plt.subplots(figsize=(150, 30), dpi=150)
#
#     plt.plot(x,y,marker='x', label ='$X$')
#     plt.xlabel('Displacement (mm)')
#     plt.ylabel('Force(N)')
#
#
#
#     plt.title('Load Protocol')
#     plt.plot(x1, y1, marker='*', label ='$Y$' )
#
#     plt.legend(title=f'{os.path.split(i)[1].split(".")[0]}')
#     plt.show()
#--------------------------
df0=pd.read_csv(paths[0])
# df1=pd.read_csv(paths[1])
# df2=pd.read_csv(paths[2])
# # df3=pd.read_csv(paths[3])
# # # #
y0 = df0['correct_force_X(N)']
y01 = df0['correct_force_Y(N)']
x0_ = df0.axis_0_2
x01_ = df0.axis_1_3
x0 = pd.concat([pd.Series([0]), x0_[:-1]], ignore_index=True)
x01 = pd.concat([pd.Series([0]), x01_[:-1]], ignore_index=True)
fig, ax = plt.subplots(figsize=(150, 30), dpi=300)
plt.rcParams['font.size'] = 5
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['lines.markersize'] = 5
#
plt.plot(x0, y0, color='k',  marker='H', markersize=3, markerfacecolor='r', label=f'{os.path.split(paths[0])[1].split(".")[0]} - $X$')
# #plt.plot(x01, y01, color='y', line = '–', marker='*',  label=f'{os.path.split(paths[0])[1].split(".")[0]} - $Y$')
plt.plot(x01, y01, color='r', marker='p', markersize=3, markerfacecolor='y', label=f'{os.path.split(paths[0])[1].split(".")[0]} - $Y$')
#
# y1 = df1['correct_force_X(N)']
# y11 = df1['correct_force_Y(N)']
# x1_ = df1.axis_0_2
# x11_ = df1.axis_1_3
# x1 = pd.concat([pd.Series([0]), x1_[:-1]], ignore_index=True)
# x11 = pd.concat([pd.Series([0]), x11_[:-1]], ignore_index=True)
#
# plt.plot(x1, y1, color='blue', marker='2', markersize=5, label=f'{os.path.split(paths[1])[1].split(".")[0]} - $X$')
#plt.plot(x11, y11,сolor='blue',  marker='d',line='–', label=f'{os.path.split(paths[1])[1].split(".")[0]} - $Y$')
# plt.plot(x11, y11, color='blue', marker='1', markersize=5, label=f'{os.path.split(paths[1])[1].split(".")[0]} - $Y$')
# # #
# y2 = df2['correct_force_X(N)']
# y21 = df2['correct_force_Y(N)']
# x2_ = df2.axis_0_2
# x21_ = df2.axis_1_3
# x2 = pd.concat([pd.Series([0]), x2_[:-1]], ignore_index=True)
# x21 = pd.concat([pd.Series([0]), x21_[:-1]], ignore_index=True)
#
# plt.plot(x2, y2, color='r', marker='^', markersize=3, markerfacecolor='black', label=f'{os.path.split(paths[2])[1].split(".")[0]} - $X$')
# #plt.plot(x21, y21,сolor='green',line = '-', marker='2', label=f'{os.path.split(paths[2])[1].split(".")[0]} - $Y$')
# plt.plot(x21, y21, color='r', marker='v', markersize=3, markerfacecolor='black', label=f'{os.path.split(paths[2])[1].split(".")[0]} - $Y$')
#
# y3 = df3['correct_force_X(N)']
# y31 = df3['correct_force_Y(N)']
# x3_ = df3.axis_0_2
# x31_ = df3.axis_1_3
# x3 = pd.concat([pd.Series([0]), x3_[:-1]], ignore_index=True)
# x31 = pd.concat([pd.Series([0]), x31_[:-1]], ignore_index=True)
#
# plt.plot(x3, y3, marker='2', label=f'{os.path.split(paths[3])[1].split(".")[0]} - $X$')
# #plt.plot(x31, y31,с='red',line = '-', marker='2', label=f'{os.path.split(paths[3])[1].split(".")[0]} -  $Y$')
# plt.plot(x31, y31, marker='2', label=f'{os.path.split(paths[3])[1].split(".")[0]} -  $Y$')

#

plt.legend()
plt.yticks(fontsize=5)
plt.xticks(fontsize=5)
plt.grid(linestyle='--', linewidth=0.1)
plt.xlabel('Перемещение[мм]', fontsize=6)
plt.ylabel('Нагрузка[Н]', fontsize=6)
plt.title('Тест на изотропность', fontsize=6)
plt.show()

# top=0.919
# bottom=0.184
# left=0.077
# right=0.985
# hspace=0.2
# wspace=0.2