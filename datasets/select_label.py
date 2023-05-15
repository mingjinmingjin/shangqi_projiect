import json
import os

label = {}
scene_num={}
if __name__ == '__main__':
    with open("/ssd/share/shangqi_data/scene_mapping.json", "r") as f:
        scene_mapping = json.load(f)
    with open("/ssd/share/shangqi_data/scene_tag.json", "r") as f:
        scene_tag = json.load(f)
    scene_num={}
    with open("/ssd/share/shangqi_data/last_dir_path.txt", 'r') as f:
        dirs = f.readlines()
    for dir in dirs:
        files = os.listdir(dir.strip())
        for file in files:
            if file.endswith("scenario_records.json"):
                file_path = os.path.join(dir.strip(), file)  # 获取文件的完整路径
                with open(file_path, 'r') as f:  # 打开文件
                    json_data = json.load(f)  # 解析JSON数据
                    for i in range(len(json_data['scenario'])):
                        scene = json_data['scenario'][i]['scene_tag']
                        scene_map=scene_mapping[scene]
                        if scene_map not in scene_num:
                            scene_num[scene_map]=0
                        scene_num[scene_map]+=1
                        # if scene not in label:
                        #     label[scene] = 0
                        # label[scene] += 1

    print(1)
    # with open("/ssd/share/shangqi_data/scene_num.json", "w") as f:
    #     json.dump(scene_num, f)

    label_dict = {}
    label_dict['u_turn_protected_without_obstacles'] = "turn_back"
    label_dict['start_1st_in_green'] = "start_1st_in_green"
    label_dict['turn_left_protected_without_obstacles'] = "turn_left"
    label_dict['change_lane_left_with_rear_obstacles'] = "change_lane_left"
    label_dict['follow_and_go'] = "follow_and_go"
    label_dict['change_lane_left_without_obstacles'] = "change_lane_left"
    label_dict['change_lane_right_without_obstacles'] = "change_lane_right"
    label_dict['change_lane_right_with_front_obstacles'] = "change_lane_right"
    label_dict['follow_and_stop'] = "follow_and_stop"
    label_dict['follow_dynamic'] = "follow_dynamic"
    label_dict['turn_right_protected_without_obstacles'] = "turn_right"
    label_dict['change_lane_left_with_front_rear_obstacles'] = "change_lane_left"
    label_dict['change_lane_left_with_front_obstacles'] = "change_lane_left"
    label_dict['change_lane_right_with_rear_obstacles'] = "change_lane_right"
    label_dict['u_turn_protected_with_merge'] = "turn_back"
    label_dict['stop_1st_in_red'] = "stop_1st_in_red"
    label_dict['turn_right_without_obstacles'] = "turn_right"
    label_dict['lane_borrow_by_vehicle'] = "lane_borrow"
    label_dict['turn_right_protected_with_parallel_merge'] = "turn_right"
    label_dict['turn_right_protected_with_vertical_merge'] = "turn_right"
    label_dict['change_lane_right_with_front_rear_obstacles'] = "change_lane_right"
    label_dict['cutin'] = "cutin"
    label_dict['cutout'] = "cutout"
    label_dict['turn_left_with_parallel_merge'] = "turn_left"
    label_dict['turn_left_with_opposite_vehicles'] = "turn_left"
    label_dict['turn_left_without_obstacles'] = "turn_left"
    label_dict['turn_right_with_vertical_merge'] = "turn_right"
    label_dict['stop_1st_in_red_in_waiting_area'] = "stop_1st_in_red"
    label_dict['start_1st_in_green_in_waiting_area'] = "start_1st_in_green"
    label_dict['turn_left_protected_with_parallel_merge'] = "turn_left"
    label_dict['turn_left_with_pedestrians_bikes'] = "turn_left"
    label_dict['turn_right_with_parallel_merge'] = "turn_right"
    label_dict['turn_right_with_pedestrians_bikes'] = "turn_right"
    label_dict['lane_borrow_by_traffic_cone']="lane_borrow"
    label_dict['follow_lead']="follow_dynamic"
    label_dict['junction_turn_left']="turn_left"
    label_dict['turn_left_with_vertical_merge']="turn_left"
    label_dict["TBD"] = "follow_dynamic"
    label_dict['lane_change']='change_lane_left'
    label_tag={}
    cnt_tag=0
    for value in label_dict.values():
        if value not in label_tag:
            label_tag[value]=cnt_tag
            cnt_tag+=1
    with open("/ssd/share/shangqi_data/scene_tag.json", "w") as f:
        json.dump(label_tag, f)
    with open("/ssd/share/shangqi_data/scene_mapping.json", "w") as f:
        json.dump(label_dict, f)
    print(1)





