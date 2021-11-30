"""
CW数据集数据文件目录
CWdata中每个文件中振动数据点约122000个数据点
"""
import os

root_dir = r"E:\dataset\CWdata_12k"


def get_filename(root_path):
    file = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file]
    if len(file_list) != 1:
        print(f"There are {len(file_list)} in '{root_path}'. Please check your file!")
        print('exit.')
        exit()
    return file_list[0]


# NC
NC = [r'NC\0', r'NC\1', r'NC\2', r'NC\3']
NC_0 = get_filename(os.path.join(root_dir, NC[0]))
NC_1 = get_filename(os.path.join(root_dir, NC[1]))
NC_2 = get_filename(os.path.join(root_dir, NC[2]))
NC_3 = get_filename(os.path.join(root_dir, NC[3]))

# IF
IF_7_file = [r'007\IF\0', r'007\IF\1', r'007\IF\2', r'007\IF\3']
IF_14_file = [r'014\IF\0', r'014\IF\1', r'014\IF\2', r'014\IF\3']
IF_21_file = [r'021\IF\0', r'021\IF\1', r'021\IF\2', r'021\IF\3']
IF_7 = [get_filename(os.path.join(root_dir, f)) for f in IF_7_file]
IF_14 = [get_filename(os.path.join(root_dir, f)) for f in IF_14_file]
IF_21 = [get_filename(os.path.join(root_dir, f)) for f in IF_21_file]

# OF
OF_7_file = [r'007\OF\0', r'007\OF\1', r'007\OF\2', r'007\OF\3']
OF_14_file = [r'014\OF\0', r'014\OF\1', r'014\OF\2', r'014\OF\3']
OF_21_file = [r'021\OF\0', r'021\OF\1', r'021\OF\2', r'021\OF\3']
OF_7 = [get_filename(os.path.join(root_dir, f)) for f in OF_7_file]
OF_14 = [get_filename(os.path.join(root_dir, f)) for f in OF_14_file]
OF_21 = [get_filename(os.path.join(root_dir, f)) for f in OF_21_file]

# RF
RF_7_file = [r'007\RF\0', r'007\RF\1', r'007\RF\2', r'007\RF\3']
RF_14_file = [r'014\RF\0', r'014\RF\1', r'014\RF\2', r'014\RF\3']
RF_21_file = [r'021\RF\0', r'021\RF\1', r'021\RF\2', r'021\RF\3']
RF_7 = [get_filename(os.path.join(root_dir, f)) for f in RF_7_file]
RF_14 = [get_filename(os.path.join(root_dir, f)) for f in RF_14_file]
RF_21 = [get_filename(os.path.join(root_dir, f)) for f in RF_21_file]

#
cw_0 = [NC_0, IF_7[0], IF_14[0], IF_21[0], OF_7[0], OF_14[0], OF_21[0], RF_7[0], RF_14[0], RF_21[0]]
cw_1 = [NC_1, IF_7[1], IF_14[1], IF_21[1], OF_7[1], OF_14[1], OF_21[1], RF_7[1], RF_14[1], RF_21[1]]
cw_2 = [NC_2, IF_7[2], IF_14[2], IF_21[2], OF_7[2], OF_14[2], OF_21[2], RF_7[2], RF_14[2], RF_21[2]]
cw_3 = [NC_3, IF_7[3], IF_14[3], IF_21[3], OF_7[3], OF_14[3], OF_21[3], RF_7[3], RF_14[3], RF_21[3]]


if __name__ == '__main__':
    pass
