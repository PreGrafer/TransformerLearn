import opencc

# 输入文件和输出文件的路径
input_file_path = "cmn.txt"
output_file_path = "output.txt"

# 创建OpenCC实例，选择繁体中文转换为简体中文
converter = opencc.OpenCC("D:/anaconda3/envs/Py39/Lib/site-packages/opencc/config/t2s")

# 打开输入文件和输出文件
with open(input_file_path, "r", encoding="utf-8") as input_file:
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        # 逐行读取输入文件，转换为简体中文，然后写入输出文件
        for line in input_file:
            simplified_line = converter.convert(line)
            output_file.write(simplified_line)

print("繁体中文已成功转换为简体中文，输出文件路径为:", output_file_path)
