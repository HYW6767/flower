#划分数据集



import shutil
import random
from pathlib import Path
import set_seed

set_seed.seedset(325)


def split(dir:str,target_dir:str,proportion:float):
    target=Path(target_dir)
    file_dir=Path(dir)
    target.mkdir(parents=True,exist_ok=True)
    for p1 in file_dir.iterdir():
        new=p1.parts[-1]
        new_folder=Path(target_dir)/new
        new_folder.mkdir(parents=True,exist_ok=True)
        file_all=[]
        file_select=[]
        for p2 in p1.rglob("*"):
            file_all.append(p2)
        select_number=int(proportion*len(file_all))
        is_select=random.sample(range(len(file_all)),select_number)
        for i in is_select:
            file_select.append(file_all[i])
        for p in file_select:
            p_root=p.parts
            proportion_root=Path(target_dir)/p_root[-2]
            proportion_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p),str(proportion_root))
