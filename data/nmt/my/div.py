import random

def split_file_randomly(file_path, ratio, part1, part2):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 随机打乱行的顺序
    random.shuffle(lines)

    # 计算分割点
    split_point = int(len(lines) * ratio)

    # 分割成两个列表
    part1_lines = lines[:split_point]
    part2_lines = lines[split_point:]

    # 写入两个新的文件
    with open(part1, 'w', encoding='utf-8') as file:
        file.writelines(part1_lines)
    with open(part2, 'w', encoding='utf-8') as file:
        file.writelines(part2_lines)

# 使用函数
split_file_randomly('cmn.txt', 0.95, 'train.txt', 'no.txt')
split_file_randomly('no.txt',0.5, 'dev.txt','test.txt')

