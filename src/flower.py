from data_clean import scan_broken,quarantine_broken,scan_duplicates
from torchvision import transforms,datasets
from PIL import Image, ImageOps
import torch.nn as nn
import torch
import set_seed
from split import split
from torch.utils.data import DataLoader
from data_augment import build_loaders, MEAN, STD, SIZE
import matplotlib.pyplot as plt
from pathlib import Path
import time



set_seed.seedset(325)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备：{DEVICE}")



#损坏/错误数据隔离
bad=scan_broken("data")
# print(bad)
# print(len(bad))
quarantine_broken(bad,"data")
#数了一下数据数量对得上
# print(scan_broken("data"))
#输出发现已经没有损坏/错误数据了
# print(scan_duplicates("data/train"))
# print(scan_duplicates("data/test"))
# print(scan_duplicates("data"))
#构造预处理配置时ai写的
def build_pre(size, mean, std):
    """构造推理用的预处理配置"""
    return {
        # ---------- 尺寸 ----------
        "size": int(size),
        "resize_mode": "resize_short_then_center_crop",   # 对应 Resize + CenterCrop
        "resize_size": int(size),
        # 如果是 Resize(int(size*256/224)) + CenterCrop(size)：
        # "resize_size": int(size * 256 / 224),
        # ---------- 数值 ----------
        "mean": [float(x) for x in mean],
        "std":  [float(x) for x in std],
        "scale_to_01": True,        # ToTensor() 做过 /255
        # ---------- 颜色 / 读图 ----------
        "color_mode": "RGB",
        "exif_transpose": True,     # 对应 ImageOps.exif_transpose
        # ---------- 通道顺序 ----------
        "channel_order": "RGB",     # PIL 是 RGB；如果用过 OpenCV 要写 BGR
        "tensor_layout": "CHW",     # ToTensor 后的顺序
        # ---------- 插值 ----------
        "interpolation": "bilinear",  # torchvision Resize 默认 BILINEAR
        # ---------- 归一化后范围（调试用） ----------
        "normalized_range": "mean0_std1",
    }
pre=build_pre(SIZE,MEAN,STD)
#数据集划分(只能运行一次)，运行多了训练集会变少很多（我懒了没做检测OVO）
# split("data/train","data/val",0.2)
# print(scan_duplicates("data"))
#测试了一下目录里未出现重复文件

#数据增强
train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders()

def save_bundle(path,model,pre,classnames,metrics=None):
    torch.save({
        "bundle_version":VERSION,
        "model_class":type(model).__name__,
        "preprocess": pre,
        "num_classes": 5,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},

        
        "task":"multiclass_classification",

        "class_names": classnames, 
        "metrics": metrics or {},

    },path)
    print("已保存模型：",path)


#模型设置
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.conv1 =nn.Sequential(nn.Conv2d(3,96,11,stride=4,padding=2),
                                  nn.BatchNorm2d(96),
                                  nn.ReLU(inplace=True))
        self.conv2 =nn.Sequential(nn.Conv2d(96,256,5,padding=2),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(256,384,3,padding=1),
                                   nn.BatchNorm2d(384),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384,384,3,padding=1),
                                   nn.BatchNorm2d(384),nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(384,256,3,padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3,stride=2))
        
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(6*6*256,4096),
                                nn.ReLU(inplace=True),
                                nn.Linear(4096,4096),
                                nn.ReLU(inplace=True),
                                nn.Linear(4096,5))

    def forward(self, x):
        x=self.conv1(x)
        x=self.pool(x)
        x=self.conv2(x)
        x=self.pool(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.pool(x)
        x=self.fc(x)
        return x

#超参数设置
model=Mymodel().to(DEVICE)
EPOCHS=60
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
fn=nn.CrossEntropyLoss()
VERSION=1.0
def train():

    #计时器
    t0 = time.time()


    #训练
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   
    plt.rcParams["axes.unicode_minus"] = False   


    PATIENCE=10
    wait=0
    loss_log_train=[]
    acc_log_train=[]
    loss_log_val=[]
    acc_log_val=[]
    best_acc=0
    best_state=None
    for epoch in range(1,EPOCHS+1):
        model.train()
        if epoch==1:
            print("训练开始")
        total_loss = correct = total = 0
        for x,y in train_loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            pred=model(x)
            loss=fn(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()* y.size(0)
            total+=y.size(0)
            correct+=(pred.argmax(1) == y).sum().item()
        loss_log_train.append(total_loss/total)
        acc_log_train.append(correct/total)
        if epoch%10==0:
            print(f"轮数：{epoch}，loss：{total_loss/total}")
        model.eval()
        with torch.no_grad():
            total_loss = correct = total = 0
            for x,y in val_loader:
                x,y=x.to(DEVICE),y.to(DEVICE)
                pred=model(x)
                loss=fn(pred,y)
                total_loss+=loss.item()* y.size(0)
                total+=y.size(0)
                correct+=(pred.argmax(1) == y).sum().item()
            
            loss_log_val.append(total_loss/total)
            acc_log_val.append(correct/total)
            if acc_log_val[-1]>best_acc:
                best_acc=acc_log_val[-1]
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                wait=0
            else:
                wait+=1
            if wait==PATIENCE:
                print(f"早停：模型已连续{PATIENCE}轮未进步")
                break


    if best_state is not None:
        model.load_state_dict(best_state)

    dt = time.time() - t0
    print(f"训练耗时 {dt:.2f}秒，最终最佳确率：{best_acc:.2f}")
    fig, axes = plt.subplots(1,2,figsize=(8, 5))
    axes[0].plot(loss_log_train)
    axes[0].set_title("Train Loss")
    axes[1].plot(acc_log_train)
    axes[1].set_title("Train Accuracy")
    plt.savefig("docs/train_curve.png")
    plt.show()
    fig, axes = plt.subplots(1,2,figsize=(8, 5))
    axes[0].plot(loss_log_val)
    axes[0].set_title("Val Loss")
    axes[1].plot(acc_log_val)
    axes[1].set_title("Val Accuracy")
    plt.savefig("docs/val_curve.png")
    plt.show()
    model.eval()
    with torch.no_grad():
        total_loss = correct = total = 0
        for x,y in test_loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            pred=model(x)
            loss=fn(pred,y)
            total_loss+=loss.item()* y.size(0)
            total+=y.size(0)
            correct+=(pred.argmax(1) == y).sum().item()
        acc_test=correct/total
    print(f"测试集准确率{acc_test}")

        
    save_bundle("src/flower.pt",model,pre,train_ds.classes)


if __name__ == "__main__":
    train()

class FlowerPredictor():
    def __init__(self,path,device="cuda"):
        b=torch.load(path,map_location=device)
        if b["bundle_version"] != VERSION:
            raise RuntimeError(f"版本不匹配：文件 v{b['bundle_version']}，代码 v{VERSION}")

        self.classes=b["class_names"]
        self.pre=b["preprocess"]
        self.device=torch.device(device)
        self.model=Mymodel()
        self.model.load_state_dict(b["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
    @torch.no_grad()
    def pred(self,img):
        if self.pre.get("exif_transpose", True):
            img = ImageOps.exif_transpose(img)
        img = img.convert(self.pre.get("color_mode", "RGB"))
        tf = transforms.Compose([
        transforms.Resize(self.pre["size"]),
        transforms.CenterCrop(self.pre["size"]),
        transforms.ToTensor(),
        transforms.Normalize(self.pre["mean"], self.pre["std"]),
        ])
        x = tf(img).unsqueeze(0).to(self.device)
        self.model.eval()
        logits = self.model(x)
        prob = torch.softmax(logits, dim=1)[0]
        conf, idx = prob.max(dim=0)
        return self.classes[idx],conf