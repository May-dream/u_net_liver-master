from torch.utils.data import Dataset
import PIL.Image as Image
import os


def make_dataset(root):
    imgs=[]
    n=len(os.listdir(root))//2
    for i in range(n):
        # 组合成路径的形式
        img=os.path.join(root,"%03d.png"%i)
        mask=os.path.join(root,"%03d_mask.png"%i)
        imgs.append((img,mask))         #append只能有一个参数，加上[]变成一个list
    return imgs


# liver_dataset = LiverDataset("data/train",transform=x_transforms,target_transform=y_transforms)
# 在Pytorch 中，数据加载可以通过自己定义的数据集对象来实现。数据集对象被抽象为Dataset类，实现自己定义的数据集需要继承Dataset,并实现两个Python魔法方法。
# __getitem__: 返回一条数据或一个样本。obj[index]等价于obj.__getitem__(index).
# __len__: 返回样本的数量。len(obj)等价于obj.__len__().

class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    # 凡是在类中定义了这个__getitem__ 方法，那么它的实例对象（假定为p），可以像这样
    # p[key] 取值，当实例对象做p[key] 运算时，会调用类中的方法__getitem__。
    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
