import os
import json
def get_last_level_folders(folder_path):
    """
    获取指定文件夹下所有最后一级文件夹的路径
    """
    last_level_folders = []
    for root, dirs, files in os.walk(folder_path):
        if not dirs and os.path.exists(os.path.join(root,"bagInfo.json")):  # 当前目录下没有子文件夹，说明是最后一级文件夹
            bagInfo_path = os.path.join(root, 'bagInfo.json')  # 先打开bagInfo文件，提取出开始时间，持续时间
            with open(bagInfo_path) as f:
                json_data = json.load(f)
                start = json_data['start']
                duration1 = json_data['duration']
            for file in files:
                if file.endswith('scenario_records.json'):
                    file_path = os.path.join(root, file)  # 获取文件的完整路径
                    with open(file_path, 'r') as f:  # 打开文件
                        json_data = json.load(f)  # 解析JSON数据
                        # scene_end = json_data['scenario'][0]['tN'] - start
                        if len(json_data['scenario'])>0 and json_data['scenario'][0]['tN'] - start>0:
                           last_level_folders.append(root)
    return last_level_folders

def save_to_txt(file_path, data):
    """
    将数据保存到指定的 txt 文件中
    """
    with open(file_path, 'w') as f:
        for item in data:
            f.write(item + '\n')

if __name__ == '__main__':
    folder_path = '/ssd/share/shangqi_data'
    file_path = '/ssd/share/shangqi_data/last_dir_path.txt'
    last_level_folders = get_last_level_folders(folder_path)
    save_to_txt(file_path, last_level_folders)