import visdom
import torch

# 2019年08月02日20:53:31
# vis 有line 类似于plot   image 可视化图片   text 记录日志合文字等    scatter bar 柱状 pie
# 支持 numpy 和 Tensor
vis = visdom.Visdom(env='test_1')

x = torch.arange(1, 30, 0.01)
y = torch.sin(x)
vis.line(X=x, Y=y, win='sinx', opts={'title': 'y=sin(x)'})

# 追加数据
for ii in range(0,10):
    x = torch.tensor([ii])
    y = x
    vis.line(X=x, Y=y, win='polynomial', name= "this is trace",update='append')
# win 窗口决定是否是同一个口，name决定线条的颜色
x = torch.arange(0,9,0.1)
y = (x **2 )/9
vis.line(X=x, Y=y, win='polynomial', name="this is trace2", update='append')

# 可视化黑白 没有s
# vis.image(torch.randn(64, 64).numpy() , win= "random2")
vis.images(torch.randn(36, 1 ,64, 64).numpy() , win= "random2")
# 可视化彩色  有 S
vis.images(torch.randn(36, 3, 64, 64).numpy(), nrow=6, win= "random3",opts={'title ':'random_imgs'})

# 文本化
vis.text(u'''<h1> asdasdasda</h1>
                <br> asdfdsfasfasdfasafaaafadfasfasdfafafasfaasfasfdfasf''',win='text' , opts={'title':u'visdom asdasdas'})