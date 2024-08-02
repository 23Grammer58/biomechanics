import meshio
import numpy as np
import os
import subprocess


def add_dir_to_path(directory_name, filename, post_fix):
    return os.path.join(directory_name, filename + post_fix)


def check(number):
    print(number)
    if number & 0x8 != 0:
        print("условие закрепления")
    if number & 0x1 != 0:
        print("x - координата неподвижна")
    if number & 0x2 != 0:
        print("y - координата неподвижна")
    if number & 0x4 != 0:
        print("z - координата неподвижна")


def replace_vtktypeint(mesh_file, postfix=".vtk"):
    file_path = add_dir_to_path("meshes", mesh_file, postfix)
    with open(file_path, "r") as file:
        lines = file.readlines()

    with open(file_path, "w") as file:
        for line in lines:
            modified_line = line.replace("vtktypeint32", "int")
            modified_line = line.replace("vtktypeint64", "int")
            file.write(modified_line)


def add_u_bc(ic, bc, point_type):

    v_x = ic
    v_bnd = bc

    v_bnd_14 = v_bnd == point_type
    # v_bnd_13 = v_bnd == 13

    v_x_bc_14_idx = np.unique(np.nonzero(v_bnd_14 * v_x)[0])
    print(v_x_bc_14_idx)
    print(v_x)

    for index in v_x_bc_14_idx:
        v_x[index][1] += dx
    return ic


def next_quasi_static(mesh_filename="solution_HOG_01"):

    it = int(mesh_filename[-2:])
    directory = r"meshes"

    # path_to_current_file = os.path.join(directory, mesh_filename + "_txt.vtk")
    path_to_current_file = add_dir_to_path(directory_name="meshes", filename=mesh_filename, post_fix="_txt.vtk")

    mesh = meshio.read(path_to_current_file)
    print("old mesh", mesh)
    v_x = mesh.point_data['v:x']
    v_bnd = mesh.point_data['v:bnd']

    du = 0.03
    dv = 0.03
    dx = np.array([du, 0, 0])
    dy = np.array([0, dv, 0])

    keys_bc = {"dy": [131, 132], "dx": [141, 142]}

    v_x = mesh.points.copy()

    v_bnd_142 = np.array(v_bnd == 142)
    v_bnd_141 = np.array(v_bnd == 141)

    v_bnd_132 = np.array(v_bnd == 132)
    v_bnd_131 = np.array(v_bnd == 131)

    if len(np.shape(v_bnd_131)) < 2:
        v_bnd_131 = v_bnd_131[:, np.newaxis]
        v_bnd_141 = v_bnd_141[:, np.newaxis]
        v_bnd_132 = v_bnd_132[:, np.newaxis]
        v_bnd_142 = v_bnd_142[:, np.newaxis]

    v_x_bc_142_idx = np.unique(np.nonzero(v_bnd_142 * v_x)[0])
    v_x_bc_141_idx = np.unique(np.nonzero(v_bnd_141 * v_x)[0])
    v_x_bc_132_idx = np.unique(np.nonzero(v_bnd_132 * v_x)[0])
    v_x_bc_131_idx = np.unique(np.nonzero(v_bnd_131 * v_x)[0])

    for index in v_x_bc_142_idx:
        v_x[index] += dx

    for index in v_x_bc_132_idx:
        v_x[index] += dy

    for index in v_x_bc_141_idx:
        v_x[index] -= dx

    for index in v_x_bc_131_idx:
        v_x[index] -= dy
        # print(mesh.points)
    # print(mesh.point_data['v:x'])

    mesh.points = mesh.point_data['v:x'].copy()
    # mesh.point_data['v:x'] = v_x
    # print("new data")
    # print(mesh.points)
    # print(mesh.point_data['v:x'])

    it += 1
    mesh_filename = mesh_filename[:-2] + str(it).zfill(2)
    path_to_next_file = os.path.join(directory, mesh_filename + "_txt.vtk")
    print(f"Results writed to {path_to_next_file}")
    meshio.write(path_to_next_file, mesh, binary=True)

    print(mesh)
    replace_vtktypeint(mesh_filename)
    return mesh_filename


def read_msh_write_vtk(mesh_filename: str, output_mesh_filename: str, print_bnd_data: bool = False):

    path_to_current_file = add_dir_to_path("meshes", mesh_filename, ".msh")
    mesh = meshio.read(path_to_current_file)

    # print(mesh.point_data)
    vertex_bnd_msh = mesh.cells_dict['vertex'][:, 0]
    bnd_idx_msh = mesh.cell_data_dict['gmsh:physical']['vertex']

    if print_bnd_data:
        print('vertex_bnd from msh')
        print(vertex_bnd_msh)
        print('bnd_idx msh')
        print(bnd_idx_msh)
        print('mesh cells')
        print(mesh.cells_dict['triangle'])

    v_bnd = []
    n1 = 0
    n2 = 0
    end = False
    try:
        while not end
