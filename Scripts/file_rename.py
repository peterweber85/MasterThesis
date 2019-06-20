
import os

ROOT_PATH = os.getenv('GDRIVE_FOLDER') + "MFP - Satellogic/images/revised_images_size512_baseres1m_ext"
categories = ["forest-woodland", "agriculture", "shrubland-grassland", "semi-desert"]
labels = [0, 1, 2]
NAME_FILTER = "size512_baseres1m"


def add_extension(folder, name_filter, extension='.png'):
    files_renamed_count = 0
    for file_name in os.listdir(folder):
        if name_filter in file_name:
            file_path_init = os.path.join(folder, file_name)
            file_path_new = os.path.join(folder, file_name+extension)
            os.rename(file_path_init, file_path_new)
            files_renamed_count += 1
    print(folder + ": " + str(files_renamed_count))


for category in categories:
    print("Renaming files of category " + category + " ...")
    folder_path_category = os.path.join(ROOT_PATH, category)
    add_extension(folder_path_category, name_filter=NAME_FILTER)
    for label in labels:
        folder_path_label = os.path.join(folder_path_category, "label_" + str(label))
        add_extension(folder_path_label, name_filter=NAME_FILTER)
    print()

print()
print("DONE!")
print()
