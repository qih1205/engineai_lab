from pathlib import Path


# 获取当前文件所在目录的父目录 (即 engineai_lab 扩展的根目录)
ENGINEAI_LAB_EXT_DIR = Path(__file__).resolve().parent.parent.as_posix()