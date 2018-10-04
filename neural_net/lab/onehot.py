# onehot.py

# Importing dependencies
import numpy as np

# Code
class onehot:
    def __init__(self, options):
        self.opt2num = {}
        self.num2opt = {}
        for opt in options:
            if not opt in self.opt2num:
                self.num2opt[len(self.opt2num)] = opt
                self.opt2num[opt] = len(self.opt2num)
        self.n = len(self.opt2num)
        self.lib = {}
    def hot_opt(self, opt):
        tmp = np.zeros(self.n)
        tmp[self.num(opt)] = 1
        return tmp
    def hot_num(self, num):
        tmp = np.zeros(self.n)
        tmp[num] = 1
        return tmp
    def opt(self, num):
        return self.num2opt[num]
    def num(self, opt):
        return self.opt2num[opt]
