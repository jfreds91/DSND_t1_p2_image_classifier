# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:29:17 2019

@author: Jesse
"""

support_file = open('supported_models.txt', 'r')
supported_models = support_file.readlines()
supported_models = [s.strip() for s in supported_models]
support_file.close()

print(supported_models)

x = 'alexnet' in supported_models
print(x)