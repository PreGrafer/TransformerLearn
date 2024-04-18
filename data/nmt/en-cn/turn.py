def process_dataset(file_path, output_path):
    with open(file_path, 'r',encoding='utf-8') as f:
        lines = f.read().split("--\n")
    result = []
    for line in lines:
        entries = line.split("\n")
        if(entries.__len__() >= 2):
            english = entries[0].split(': ')
            mandarin = entries[2].split(': ')
            if english[0] == 'english' and mandarin[0]== 'mandarin':
                result.append(f"{english[1].strip()}\t{mandarin[1].strip()}")
    with open(output_path, 'w',encoding='utf-8') as f:
        f.write("\n".join(result))

# 使用方法
process_dataset('hsk_1_4.txt', 'hsk.txt')
