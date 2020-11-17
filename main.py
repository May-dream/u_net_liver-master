import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms

from mino import get_iou
from unet import Unet
from dataset import LiverDataset


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 指定使用那一块显卡
# torch.cuda.set_device(0)

# PyTorch框架中有一个非常重要且好用的包：torchvision，
# 该包主要由3个子包组成，分别是：torchvision.datasets、torchvision.models、torchvision.transforms

# transforms.ToTensor()将一个取值范围是[0,255]的PLT.Image或者shape为(H,W,C)的NUMPY.ndarray,
# 转换成形式为(C,H,W)，取值范围在(0,1)的torch.floatTensor
# 而后面的transform.Normalize()则把0-1变换到(-1,1)
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

# 设置epochs的数量为20
def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 反向传播，参数优化
            loss.backward()
            optimizer.step()
            # 计算LOSS总和
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))

    # 保存model
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model

#训练模型
def train(args):
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size                # 命令行中获取batch_size的大小
    criterion = nn.BCEWithLogitsLoss()          # 创建一个标准来度量目标和输出之间的二进制交叉熵。
    # 实际中我们可以使用 Adam 作为默认的优化算法，往往能够达到比较好的效果
    # 梯度下降，寻找极值点使LOSS最小
    optimizer = optim.Adam(model.parameters())

    # 初始化
    liver_dataset = LiverDataset("data/train",transform=x_transforms,target_transform=y_transforms)

    # 读取数据集
    # 它为我们提供的常用操作有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作), num_workers(加载数据的时候使用几个子进程)
    # shuffle是否将数据打乱；
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)

#显示模型的输出结果
def test(args):
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    liver_dataset = LiverDataset("data/val", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()

    # import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y=model(x).sigmoid()
            img_y=torch.squeeze(y).numpy()

            # get_iou("data/val/000_mask.png",img_y)
            plt.imshow(img_y)
            plt.pause(0.1)
        plt.show()


if __name__ == '__main__':
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)
