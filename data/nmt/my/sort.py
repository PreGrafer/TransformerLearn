def sort_file(file_path,write_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 按照英文段的长度、首字母和中文段的长度排序
    sorted_lines = sorted(lines,
                          key=lambda line: (len(line.split('\t')[0]), line.split('\t')[0], len(line.split('\t')[1])))

    # 写入排序后的内容
    with open(write_path, 'w', encoding='utf-8') as file:
        file.writelines(sorted_lines)

# 使用函数
sort_file('train.txt', 'cmn.txt')
