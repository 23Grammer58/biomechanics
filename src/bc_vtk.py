import vtk


def read_vtk(file_path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()


def process_vtk(file_path, dy):
    grid = read_vtk(file_path)

    points = grid.GetPoints()
    num_points = points.GetNumberOfPoints()

    max_y = -float('inf')
    min_y = float('inf')
    max_y_points = []
    min_y_points = []

    for i in range(num_points):
        x, y, z = points.GetPoint(i)
        if y > max_y:
            max_y = y
            max_y_points = [i]
        elif y == max_y:
            max_y_points.append(i)
        if y < min_y:
            min_y = y
            min_y_points = [i]
        elif y == min_y:
            min_y_points.append(i)

    new_points = vtk.vtkPoints()
    boundary_tags = vtk.vtkIntArray()
    boundary_tags.SetName("v:bnd")

    updated_coords = vtk.vtkDoubleArray()
    updated_coords.SetName("v:x")
    updated_coords.SetNumberOfComponents(3)
    updated_coords.SetNumberOfTuples(num_points)

    for i in range(num_points):
        x, y, z = points.GetPoint(i)
        new_points.InsertNextPoint(x, y, 0.0)  # Zeroing the z-coordinate
        if i in max_y_points:
            updated_coords.SetTuple3(i, x, y + dy, 0.0)  # Zeroing the z-coordinate
            boundary_tags.InsertNextValue(15)
        elif i in min_y_points:
            updated_coords.SetTuple3(i, x, y - dy, 0.0)  # Zeroing the z-coordinate
            boundary_tags.InsertNextValue(15)
        else:
            updated_coords.SetTuple3(i, x, y, 0.0)  # Zeroing the z-coordinate
            boundary_tags.InsertNextValue(4)

    grid.SetPoints(new_points)
    grid.GetPointData().AddArray(boundary_tags)
    grid.GetPointData().AddArray(updated_coords)

    # Filter out non-triangle cells
    triangle_cells = vtk.vtkCellArray()
    cell_types = vtk.vtkUnsignedCharArray()
    cell_types.SetNumberOfComponents(1)

    for i in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:
            triangle_cells.InsertNextCell(cell)
            cell_types.InsertNextValue(vtk.VTK_TRIANGLE)

    grid.SetCells(cell_types, triangle_cells)

    # Create a new cell data array for f:thickness
    thickness_data = vtk.vtkFloatArray()
    thickness_data.SetName("f:thickness")
    thickness_data.SetNumberOfComponents(1)
    thickness_data.SetNumberOfTuples(grid.GetNumberOfCells())

    for i in range(grid.GetNumberOfCells()):
        thickness_data.SetValue(i, 1.0)

    grid.GetCellData().AddArray(thickness_data)

    return grid


def write_vtk(grid, output_path):
    num_points = grid.GetNumberOfPoints()
    num_cells = grid.GetNumberOfCells()

    with open(output_path, 'w') as file:
        file.write('# vtk DataFile Version 3.0\n')
        file.write('File written by membrane-model\n')
        file.write('ASCII\n')
        file.write('DATASET UNSTRUCTURED_GRID\n')

        points = grid.GetPoints()
        file.write(f'POINTS {num_points} float\n')
        for i in range(num_points):
            x, y, z = points.GetPoint(i)
            file.write(f'{x} {y} 0.0\n')  # Zeroing the z-coordinate

        cells = grid.GetCells()
        cell_array = cells.GetData()
        file.write(f'CELLS {num_cells} {cell_array.GetNumberOfTuples()}\n')
        cell_id_list = vtk.vtkIdList()
        for i in range(num_cells):
            grid.GetCellPoints(i, cell_id_list)
            num_cell_points = cell_id_list.GetNumberOfIds()
            file.write(f'{num_cell_points}')
            for j in range(num_cell_points):
                file.write(f' {cell_id_list.GetId(j)}')
            file.write('\n')

        cell_types = grid.GetCellTypesArray()
        file.write(f'CELL_TYPES {num_cells}\n')
        for i in range(num_cells):
            file.write(f'{cell_types.GetValue(i)}\n')

        point_data = grid.GetPointData()
        file.write(f'POINT_DATA {num_points}\n')

        v_bnd = point_data.GetArray('v:bnd')
        file.write('SCALARS v:bnd int 1\n')
        file.write('LOOKUP_TABLE default\n')
        for i in range(num_points):
            file.write(f'{v_bnd.GetValue(i)}\n')

        v_x = point_data.GetArray('v:x')
        file.write('SCALARS v:x double 3\n')
        file.write('LOOKUP_TABLE default\n')
        for i in range(num_points):
            file.write(f'{v_x.GetComponent(i, 0)} {v_x.GetComponent(i, 1)} 0.0\n')  # Zeroing the z-coordinate

        cell_data = grid.GetCellData()
        file.write(f'CELL_DATA {num_cells}\n')

        f_thickness = cell_data.GetArray('f:thickness')
        file.write('SCALARS f:thickness float 1\n')
        file.write('LOOKUP_TABLE default\n')
        for i in range(num_cells):
            file.write(f'{f_thickness.GetValue(i)}\n')


if __name__ =="__main__":
    input_vtk_file = r'C:\Users\User\Documents\projects\study\meshes_course\gmsh\patch_with_all_tags.vtk'
    output_vtk_file = r'C:\Users\User\PycharmProjects\pythonProject\meshes\biomech_school\test_rake.vtk'
    dy = 5.0 # Example value for dy
    grid = process_vtk(input_vtk_file, dy)
    write_vtk(grid, output_vtk_file)

