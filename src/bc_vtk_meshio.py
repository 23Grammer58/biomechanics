import meshio
import numpy as np
import os
import subprocess
from bc_vtk import write_vtk, read_vtk

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

    path_to_current_file = add_dir_to_path("..\meshes", mesh_filename, ".msh")
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
        while not end:
            if n2 == vertex_bnd_msh[n1]:
                v_bnd.append([bnd_idx_msh[n1]])
                n2 += 1
                n1 += 1
            else:
                v_bnd.append([4])
                n2 += 1
    except IndexError:
        end = True

        others_vertex_bnd_edx = np.ones(len(mesh.points) - len(v_bnd))[:, np.newaxis] * 4
        v_bnd = np.array(v_bnd)

        # print(v_bnd)
        # print(others_vertex_bnd_edx)
        # print(f"shape v_bnd = {np.shape(v_bnd)}, shape other = {np.shape(others_vertex_bnd_edx)}")

        v_bnd = np.concatenate((v_bnd, others_vertex_bnd_edx))

    assert len(v_bnd) == len(mesh.points)
    mesh.point_data['v:bnd'] = v_bnd

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

    # if len(np.shape(v_bnd_13)) < 2:
    #     v_bnd_13 = v_bnd_13[:, np.newaxis]
    #     v_bnd_14 = v_bnd_14[:, np.newaxis]

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

    mesh.point_data['v:x'] = v_x

    # v_bnd
    v_bnd = v_bnd.flatten()
    v_bnd = np.array([int(x // 10) if x in [141, 142, 131, 132] else int(x) for x in v_bnd], dtype=int)[:, np.newaxis]
    mesh.point_data['v:bnd'] = v_bnd

    cells_vtk = [("triangle", mesh.cells[-1].data)]
    cells_num = np.shape(mesh.cells[-1].data)

    f_fiber = np.ones(cells_num) * [1, 0, 0]
    s_fiber = np.ones(cells_num) * [0, 1, 0]
    thickness = np.ones(cells_num[0]) * 0.4

    cell_data_vtk = {
        "f:fiber_s": [s_fiber],
        "f:fiber_f": [f_fiber],
        "f:thickness": [thickness[:, np.newaxis]],
    }

    # cell_data_vtk = {key: np.array([np.array([elem]) for elem in value]) for key, value in cell_data_vtk.items()}
    print(cell_data_vtk)

    mesh_vtk = meshio.Mesh(
        mesh.points,
        cells_vtk,
        point_data=mesh.point_data,
        cell_data=cell_data_vtk
    )

    assert "vertex" not in mesh_vtk.cells_dict.keys()

    path_to_reformatted_file = add_dir_to_path("..\meshes", output_mesh_filename, ".vtk")
    meshio.write(path_to_reformatted_file, mesh_vtk, binary=False)

    return mesh_vtk

def reformat_vtk(path: str):
    write_vtk(read_vtk(path), path[:-4] + "_proc.vtk")




if __name__ == "__main__":

    # rewrite()
    read_msh_write_vtk("square_with_holes_tags", "test", True)
    reformat_vtk(add_dir_to_path("../meshes", "test", ".vtk"))
    # check(13)

    # check(14)
    # it = 1
    # filename_prev = "solution_HOG_01"
    # while it < 3:
    #     filename_new = next_quasi_static(filename_prev)
    #     filename_prev = filename_new
    #
    #     it += 1

