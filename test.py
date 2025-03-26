import torch
print('torch版本：'+torch.__version__)
print('cuda是否可用：'+str(torch.cuda.is_available()))
print('cuda版本：'+str(torch.version.cuda))
print('cuda数量:'+str(torch.cuda.device_count()))
print('GPU名称：'+str(torch.cuda.get_device_name()))