#这是一个用于清洗数据的Python文件

import shutil
from pathlib import Path
import hashlib
from collections import defaultdict
import set_seed

set_seed.seedset(325)

#通过筛选文件大小寻找损坏图片
def scan_broken(dir):
    root1=Path(dir)
    bad=[]
    skip_name="quarantine"
    for root2 in root1.iterdir():
        if not root2.is_dir() or root2.name==skip_name:   # ★ 跳过隔离区
            continue
        for root3 in root2.iterdir():
            for p in root3.iterdir():
                if p.stat().st_size<=128:
                    bad.append(p)
    return bad
# broken_file=scan_broken("data")
# print(scan_broken("data"))
# print(len(scan_broken("data")))
#返回13个，刚好与ai找到的数量相同,运行完后已注释



#将损坏文件隔离
def quarantine_broken(bad,dir):#说明：bad里存储着需要隔离的文件（列表形式），dir则是创建隔离文件夹的根目录
    quarantine_dir=Path(dir)/"quarantine"
    quarantine_dir.mkdir(parents=True,exist_ok=True)
    for p in bad:
        p_root=p.parts
        quarantine_root=Path(quarantine_dir)/p_root[-3]/p_root[-2]
        quarantine_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p),str(quarantine_root))


#筛选出相同图片
#采用哈希函数进行筛选
def scan_duplicates(dir):
    file_dir=Path(dir)
    files=[]
    digests=[]
    duplicates=[]
    for p in file_dir.rglob("*"):
        if "quarantine" in str(p):
            continue
        if not p.is_file():
            continue
        files.append(p)
        with open(p,"rb") as f:
            digests.append(hashlib.file_digest(f,"sha256").hexdigest())
    a=defaultdict(list)
    for i, x in enumerate(digests):    # i 是下标，x 是值
        a[x].append(i)
    for x,idx in a.items():
        if len(idx)<2:
            continue
        duplicate=[]

        for i in idx:
            duplicate.append(files[i])
        duplicates.append(duplicate)
    return duplicates

