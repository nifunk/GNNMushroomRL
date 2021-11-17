import os
import argparse
import glob

# DESCRIPTON:
# this script is intended to reduce the size of the experimental results by removing all stored (fully saved) waypoints
# as those fully saved waypoints are super memory intesive.
# Use the script by specifying the load path, i.e. the location from where on the search for the files is started. The
# script is recursive, i.e. will search through the whole structure and always keep the latest (longest trained) waypoint
# for every folder.


def retrieve_subfolders(path):
    list_subfolders_with_paths = [[f.name, f.path] for f in os.scandir(path) if f.is_dir()]
    return list_subfolders_with_paths

def handle_folder(path):
    list_subfolders_with_paths = retrieve_subfolders(path)
    if (len(list_subfolders_with_paths) == 1):
        filter_files(list_subfolders_with_paths[0][-1])
    elif (len(list_subfolders_with_paths) > 1):
        for i in range(len(list_subfolders_with_paths)):
            handle_folder(list_subfolders_with_paths[i][-1])

    # STILL DO THE FILTERING as it could also be on this level!
    filter_files(path)

def filter_files(path):
    arr = os.listdir(path)
    max_idx_full = -100
    to_be_deleted_list = []
    max_idx_item = None
    for i in range(len(arr)):
        if (arr[i].find("agent_full")!=-1 and arr[i].find(".msh")!=-1):
            # retrive number:
            splitted = arr[i].split(".", 1)[0]
            obtain_episode = int(splitted[11:]) # remove the other strings

            if (max_idx_full==-100):
                max_idx_item = arr[i]
                max_idx_full = obtain_episode

            if (obtain_episode>max_idx_full):
                to_be_deleted_list.append(max_idx_item)
                max_idx_full = obtain_episode
                max_idx_item = arr[i]
            elif (obtain_episode<max_idx_full):
                to_be_deleted_list.append(arr[i])

    # Then remove all of them:
    for i in range(len(to_be_deleted_list)):
        os.remove(path+"/"+to_be_deleted_list[i])


parser = argparse.ArgumentParser()

parser.add_argument('--load-path', type=str, help='path to folder from where to load tensorboard data')

args = parser.parse_args()

list_subfolders_with_paths = retrieve_subfolders(args.load_path)

if (len(list_subfolders_with_paths)>1):
    for i in range(len(list_subfolders_with_paths)):
        handle_folder(list_subfolders_with_paths[i][-1])
elif (len(list_subfolders_with_paths)==1):
    filter_files(list_subfolders_with_paths[0][-1])

# also do the filtering in the current directory:
filter_files(args.load_path)

