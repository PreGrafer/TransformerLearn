import random

# 打开并读取文件
with open('hsk.txt', 'r',encoding='utf-8') as file:
    lines1 = file.readlines()

with open('train.txt', 'r',encoding='utf-8') as file:
    lines2 = file.readlines()

# 合并并随机化行
all_lines = lines1 + lines2
random.shuffle(all_lines)

# 写入新文件
with open('new_train.txt', 'w',encoding='utf-8') as file:
    file.writelines(all_lines)
