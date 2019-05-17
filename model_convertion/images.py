import os

doc = open('images.txt','w+')
path = '/home/zhex/work/DACSDC-ZXZ/Inference/data/images'
filenames = os.listdir(path)
filenames.sort(key = lambda x:int(x[0:-4]))

for filename in filenames:
    filenamex = os.path.join(path,filename)
    print(filenamex,file=doc)
