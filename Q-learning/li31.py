import os
import pandas as pd


li31_path = '../data/精度1/李31/新建文件夹'
li31_lis = os.listdir(li31_path)
li31 = pd.DataFrame()
for i in li31_lis:
    li31_i = pd.read_excel(os.path.join(li31_path,i))
    li31 = pd.concat((li31,li31_i),axis=0)
print(li31)