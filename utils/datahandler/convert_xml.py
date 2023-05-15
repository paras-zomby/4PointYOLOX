# 读取xml的相关信息
from xml.dom import minidom
import os

armor_convert_list = ["armor_infantry_4_red", "armor_infantry_4_blue", "armor_infantry_4_none",
                      "armor_hero_red", "armor_hero_blue", "armor_hero_none", "armor_sentry_red",
                      "armor_sentry_blue", "armor_sentry_none", "armor_outpost_red", "armor_outpost_blue",
                      "armor_outpost_none", "armor_base_red", "armor_base_blue", "armor_base_none",
                      "armor_engineer_red", "armor_engineer_blue", "armor_engineer_none", "armor_infantry_3_red",
                      "armor_infantry_3_blue", "armor_infantry_3_none", "armor_infantry_5_red", "armor_infantry_5_blue",
                      "armor_infantry_5_none"]


def convert_labels(xml_label_folder, save_folder):
    files = os.listdir(xml_label_folder)
    for xml_label_file in files:
        Xml = minidom.parse(os.path.join(xml_label_folder, xml_label_file))
        root = Xml.documentElement
        # 获取标签对应的图像的名字
        img_name = root.getElementsByTagName('filename')[0].childNodes[0].data
        img_name = os.path.splitext(img_name)[0]
        name = root.getElementsByTagName('name')
        bndbox = root.getElementsByTagName('bndbox')
        with open(os.path.join(save_folder, img_name) + ".txt", "w", encoding="utf-8") as f:
            for i in range(len(list(bndbox))):
                # 存储装甲板角点的坐标与ID
                cls = name[i].childNodes[0].data
                if cls in armor_convert_list:
                    cls = armor_convert_list.index(cls)
                    if i != 0:
                        f.write("\n")
                    f.write(str(cls))
                else:
                    continue
                for child in bndbox[i].childNodes:
                    if child.nodeName != '#text':
                        f.write(' ' + str(child.childNodes[0].data))


if __name__ == "__main__":
    convert_labels("../../dataset/heu_data/xml_Label", "../../dataset/heu_data/Label")
