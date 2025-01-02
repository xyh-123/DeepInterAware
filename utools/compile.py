import os
from setuptools import setup, find_packages
from Cython.Build import cythonize
import tempfile


# 读取项目主目录下的文件
def encrypt_py(path):
    filepaths = []
    '''
    加密路径下除了__init__.py的所有.py文件
    :param path:路径
    :return:
    '''
    for root, dirs, files in os.walk(path):
        for file in files:
            fullpath = os.path.join(root, file)
            print(fullpath)  # 打印完整路径

            filepath = file
            print(filepath)  # 打印文件名

            filename = os.path.splitext(file)[0]
            print(filename)  # 打印无扩展名的文件名

            if filename == "__init__":
                continue

            suffix = os.path.splitext(file)[1].split('.')[-1]
            print(suffix)

            if suffix == "py":
                filepaths.append(fullpath)
    return filepaths


# 转换为so文件（编译文件)
def py2so(root_dir, file_path):
    u"""
    @root_dir 应用的上级
    @file_path文件的路径
    把指定的py文件编译成so文件,如果so文件存在，原py文件没有改变大小，则不编译。
    """
    COMPILE_S = u"""
from distutils.core import setup
from Cython.Build import cythonize
setup(
    name = "temp",
    ext_modules = cythonize("{path}")
)"""
    if not os.path.exists(file_path):
        return False
    setup_file = COMPILE_S.format(path=file_path)
    so_file = file_path.replace(".py", ".so")
    t = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
    path = t.name
    t.write(setup_file.encode())
    t.close()
    command = "python {path} build_ext --inplace".format(path=path)
    # logger.info("command='%s'"%command)
    os.chdir(root_dir)  # 编译的时候，会在这个目录下面生成按照文件路径的so
    os.system(command)
    os.remove(path)  # 删除临时文件
    if os.path.exists(so_file):  # 编译好了so之后，删除py文件
        os.remove(file_path)
    return True


# filepaths = encrypt_py('/root/framework_finally/Abopt_so')
# for f in filepaths:
#     py2so('/root/framework_finally/Abopt_so', f)

py2so('/home/u/data/xyh/project/deepinteraware/networks', 'models.py')