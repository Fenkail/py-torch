import torch.nn as nn

class NetG(nn.Module):
    '''
    生成器定义
    '''
    # 调用同名类作为方法
    def __init__(self,opt):
        super(NetG,self).__init__()
        # 生成器 feature map 数
        ngf = opt.ngf

        # 模块将按照构造函数中传递的顺序添加到模块中
        self.main = nn.Sequential(
            # 解除卷积
            nn.ConvTranspose2d(opt.nz, ngf*8, 4 , 1 ,0 , bias=False),
            # out = （in - 1 ）×s -2 P +K
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
        #     输出形状  ngf×8  × 4 × 4

            nn.ConvTranspose2d(ngf*8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            #     输出形状  ngf×4  × 8 × 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            #     输出形状  ngf×2  × 16 × 16

            nn.ConvTranspose2d(ngf * 2, ngf , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf ),
            nn.ReLU(True),
            #     输出形状  ngf  × 32 × 32

            nn.ConvTranspose2d(ngf , 3, 5, 3, 1, bias=False),
            # 归一化 -1~1
            nn.Tanh()
        )

        def forward(self, *input):
            return self.main(input)


class NetD(nn.Module):
    '''
    判别器定义
    '''

    def __init__(self, opt):
        super(NetG).__init__()
        # 生成器 feature map 数
        ndf = opt.ndf

        # 模块将按照构造函数中传递的顺序添加到模块中
        self.main = nn.Sequential(
            # 卷积

            nn.Conv2d(ndf, 3, 5, 3, 1, bias=False),
            # inplace - 选择是否进行覆盖运算- 节省存储
            nn.LeakyReLU(0.2, inplace=True),
            #     输出形状  ngf  × 32 × 32

            nn.Conv2d(ndf , ndf*2 , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            # 经验选择 与 relu差别不大
            nn.LeakyReLU(0.2, inplace=True),
            #     输出形状  ngf×2  × 16 × 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #     输出形状  ngf×4  × 8 × 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #     输出形状  ngf×8  × 4 × 4

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # 归一化到 0~1
            nn.Sigmoid()
        )

        def forward(self, *input):
            return self.main(input).view(-1)