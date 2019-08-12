import torchvision.transforms
import torch
import logging
import visdom as vis
from tqdm import tqdm

from GAN_ import model
from torch.autograd import Variable


class Config(object):

    # 存放路径
    data_path = 'data/'
    # 进程数
    num_worker = 4
    image_size = 96
    batch_size = 256
    max_epoch = 200
    # 生成器学习率
    lr1 = 2e-4
    # 判别器学习率
    lr2 = 2e-4
    # beta1 的参数 Adam
    beta1 = 0.5
    use_gpu = True
    # 噪声维度
    nz = 100
    ngf = 64
    ndf = 64

    save_path = 'imgs/'

    vis = True
    env = 'Gan'
    # 每间隔20batch 画一出
    plot_every = 20

    debug_file = '/tmp/debuggan'
    d_every = 1
    g_every = 5
    # 十个epoch保存一次模型
    decay_every = 10
    netd_path = 'checkpoints/netd_211.pth'
    netg_path = 'checkpoints/netg_211.pth'

    # 测试时用的参数
    gen_img = 'result.png'
    # 512 生成图像中选择64张
    gen_search_num = 512
    # 噪声的均值和方差
    gen_mean = 0
    gen_std = 1
opt = Config()


# 规定了数据的格式
transforms = torchvision.transforms.Compose([
    # resize
    torchvision.transforms.Scale(opt.image_size),
    torchvision.transforms.CenterCrop(opt.image_size),
    torchvision.transforms.ToTensor(),
    # 按要求进行标准化
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
# 数据读取
dataset = torchvision.datasets.ImageFolder(opt.data_path,transforms)
# drop the last incomplete batch
dataloader = torch.utils.data.DataLoader(dataset,batch_size = opt.batch_size, shuffle = True,
                                         num_workers= opt.num_worker,drop_last =True)

# 定义网络
netg, netd = model.NetG(opt), model.NetD(opt)
# lambda 简单函数的实现方法   先加载到内存中
map_location = lambda  storage , loc:storage

# 2. cpu -> gpu 1
#
# torch.load('modelparameters.pth', map_location=lambda storage, loc: storage.cuda(1))
# 3. gpu 1 -> gpu 0
#
# torch.load('modelparameters.pth', map_location={'cuda:1':'cuda:0'})
# 4. gpu -> cpu
#
# torch.load('modelparameters.pth', map_location=lambda storage, loc: storage)
try:
    netd.load_state_dict(torch.load(opt.netd_path,map_location))
    netg.load_state_dict(torch.load(opt.netg_path,map_location))
except:
    logging.debug('debug 信息')


# 定义优化器合损失
optimizer_g = torch.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1,0.999))
optimizer_d = torch.optim.Adam(netg.parameters(), opt.lr2, betas=(opt.beta1,0.999))
#  Binary Cross Entropy 二分类交叉熵
criterion = torch.nn.BCELoss()

# 真实图片为1   假图像为0
# 发现tensor和variable输出的形式是一样的，在新版本的torch中可以直接使用tensor而不需要使用variable
true_labels = Variable(torch.ones(opt.batch_size))
fake_labels = Variable(torch.zeros(opt.batch_size))
fix_noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))
noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))

if opt.use_gpu:
    netd.cuda()
    netg.cuda()
    criterion.cuda()
    true_labels, fake_labels, = true_labels.cuda(), fake_labels.cuda()
    fix_noises, noises = fix_noises.cuda(), noises.cuda()


# 开始训练网络
# Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器
# enumerate的输出 [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
for ii, (img,_) in tqdm(enumerate(dataloader)):
    real_img = Variable(img)
    if opt.use_gpu:
        real_img = real_img.cuda()

    # 训练判别器
    if(ii+1) % opt.d_every == 0 :
        # Clear the gradients of all optimized
        optimizer_d.zero_grad()
        output = netd(real_img)
        error_d_real = criterion(output,true_labels)
        error_d_real.backward()

        # 尽可能把假图判0
        noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
        # 根据噪声生成假图
        # detach 计算图的截断
        fake_img = netg(noises).detach()
        fake_output = netd (fake_img)
        error_d_fake = criterion(fake_output,fake_labels)
        error_d_fake.backward()
        optimizer_d.step()

    # 训练生成器

    if (ii + 1) % opt.g_every == 0:
        # Clear the gradients of all optimized
        optimizer_g.zero_grad()
        noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
        fake_img = netg(noises)
        fake_output = netd (fake_img)
        # 生成器尽可能把假图判1
        error_g = criterion(fake_output, true_labels)
        error_g.backward()
        optimizer_g.step()


fix_fake_img = netg(fix_noises)
vis.images(fix_fake_img.data.cpu().numpy()[:64]*0.5 + 0.5,win= "fixfake")

