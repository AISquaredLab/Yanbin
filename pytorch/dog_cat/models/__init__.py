from .AlexNet import AlexNet
from .ResNet34 import ResNet34

'''
之后就可以写成
import models 
model = getattr('models','AlexNet')()