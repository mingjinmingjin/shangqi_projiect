import os
import torch


if __name__ == '__main__':
    files=[]
    for dirpath, dirnames, filenames in os.walk("/ssd/share/shangqi_data_processed"):
        for filename in filenames:
            # 使用os.path.join()函数连接文件夹路径和文件名，生成文件的绝对路径
            if filename.startswith("pre"): continue
            file_path = os.path.join(dirpath, filename)
            data=torch.load(file_path)
            if len(data.y)==1:
            # 将文件的绝对路径添加到files列表中
                files.append(file_path)
    with open("/ssd/share/shangqi_data/one_obj.txt", 'w') as file:
        for item in files:
            file.write(item + '\n')
    print(1)