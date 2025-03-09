def remove_n_warmup_lines(input_file, output_file):
    """
    读取输入文件，删除包含 'n_warmup' 的行，并将结果保存到输出文件中。
    
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    """
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                if 'n_warmup' not in line:
                    outfile.write(line)
        print(f"处理完成，结果已保存到 {output_file}")
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 未找到。")
    except Exception as e:
        print(f"发生错误：{e}")

# 输入文件路径和输出文件路径
input_file_path = 'result.txt'
output_file_path = 'result_cleaned.txt'

# 调用函数处理文件
remove_n_warmup_lines(input_file_path, output_file_path)
