#coding:utf8
import torch as t
import time


class BasicModule(t.nn.Module):
    #封装nn.Module 主要提供save和load
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(Itype(self))

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self,name=None):
        #默认使用名字加时间作为文件名
        if name is None:
            prefix = 'checkpoints/'+self.model_name+'_'
            name = time.strftime(prefix+'%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(),name)
        return name
        