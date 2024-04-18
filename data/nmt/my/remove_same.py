def remove_duplicate_lines_strict(file_path,out_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    english_chinese_pairs = [line.split('\t') for line in lines]
    new_pairs = []

    for i in range(len(english_chinese_pairs)):
        if i == 0 or english_chinese_pairs[i][0] != english_chinese_pairs[i-1][0]:
            new_pairs.append(english_chinese_pairs[i])

    new_lines = ['\t'.join(pair) for pair in new_pairs]

    # 写入新的内容
    with open(out_path, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)

# 使用函数
remove_duplicate_lines_strict('cmn.txt','cmn.txt')
