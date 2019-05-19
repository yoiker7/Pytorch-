import torch as t
import time

class BasicModule(t.nn.Module):
    '''封装了nn.Module，主要提供save和load两个方法'''

    def __init__(self):
        super(BasicModule, self).__init__() # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        self.module_name = str(type(self))  # 模型的默认名字

    def load(self, path):
        '''可加载指定路径的模型'''
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        '''保存模型，默认使用“模型名字+时间”作为文件名，如AlexNet_0710_23:55:29.pth'''
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')      #%H:%M:%S出问题了,把:换成_就不出错，:为什么不能用？
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class Flat(t.nn.Module):
    """
    把输入reshape成（batch_size,dim_length）
    """

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)