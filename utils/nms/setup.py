import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

filename = 'gpu_nms'  # 源文件名
full_filename = 'gpu_nms.pyx'  # 包含后缀的源文件名

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension(filename, [full_filename])],  # 配置需要cython编译的源文件
    include_dirs=[np.get_include()]
)
