import os 
import glob
import csv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

NUMPY_PREFIX = "/nfs/kun2/users/homer/datasets/bridge_data_all/numpy"
RAW_PREFIX = "/nfs/kun2/users/homer/datasets/bridge_data_all/raw"
LANG_PATHS = "/nfs/kun2/users/homer/datasets/bridge_data_all/language/rss/*/*"
FIELDNAMES = ['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward', 'CreationTime', 'MaxAssignments', 'RequesterAnnotation', 'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds', 'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds', 'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime', 'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime', 'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate', 'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Input.frame_0', 'Input.frame_1', 'Input.frame_2', 'Input.frame_3', 'Input.frame_4', 'Input.frame_5', 'Input.frame_6', 'Input.frame_7', 'Input.frame_8', 'Input.frame_9', 'Input.frame_10', 'Input.frame_11', 'Input.frame_12', 'Input.frame_13', 'Input.frame_14', 'Input.frame_15', 'Input.frame_16', 'Input.frame_17', 'Input.frame_18', 'Input.frame_19', 'Input.frame_20', 'Input.frame_21', 'Input.frame_22', 'Input.frame_23', 'Input.frame_24', 'Input.frame_25', 'Input.frame_26', 'Input.frame_27', 'Input.frame_28', 'Input.frame_29', 'Input.frame_30', 'Input.frame_31', 'Input.frame_32', 'Input.frame_33', 'Input.frame_34', 'Input.frame_35', 'Input.frame_36', 'Input.frame_37', 'Input.frame_38', 'Input.frame_39', 'Input.frame_40', 'Input.frame_41', 'Input.frame_42', 'Input.frame_43', 'Input.frame_44', 'Input.frame_45', 'Input.frame_46', 'Input.frame_47', 'Input.frame_48', 'Input.frame_49', 'Input.frame_50', 'Input.frame_51', 'Input.frame_52', 'Input.frame_53', 'Input.frame_54', 'Input.frame_55', 'Input.frame_56', 'Input.frame_57', 'Input.frame_58', 'Input.frame_59', 'Answer.task_descr', 'Approve', 'Reject']

def parse_csv_name(path):
    csv_name = path.split("/")[-1]
    words = csv_name.split("__")
    category = words[0][len("extra_bridge_dataset_"):]
    domain, task = words[1:3]
    return category, domain, task

timestamp_to_lang = defaultdict(list)

for l in glob.glob(LANG_PATHS):
    print(l)
    category, domain, task = parse_csv_name(l)
    numpy_path = os.path.join(NUMPY_PREFIX, category, domain, task)
    groups = glob.glob(os.path.join(numpy_path, "*"))
    groups.sort(key=lambda x: int(x.split("/")[-1]))
    all_traj = defaultdict(dict)
    for g in groups:
        train_path = os.path.join(numpy_path, g, "train/out.npy")
        val_path = os.path.join(numpy_path, g, "val/out.npy")
        train_data = np.load(train_path, allow_pickle=True)
        val_data = np.load(val_path, allow_pickle=True)
        group = g.split("/")[-1]
        all_traj[group]["train"] = train_data
        all_traj[group]["val"] = val_data

    total = 0
    with open(l, newline='', encoding='mac_roman') as csvfile:
        reader = csv.DictReader((line.replace('\0','') for line in csvfile), fieldnames=FIELDNAMES, dialect='excel')
        i = 0
        for row in reader:
            i += 1
            if i == 1:
                continue
            group, split, index = row['Input.frame_0'].split("/")[-5:-2]
            timestamp = all_traj[group][split][int(index)]['observations'][0]['time_stamp']
            timestamp_to_lang[timestamp].append(row['Answer.task_descr'])
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(all_traj[group][split][int(index)]['observations'][0]['images0'])
            # ax[1].imshow(all_traj[group][split][int(index)]['observations'][-1]['images0'])
            # plt.title(row['Answer.task_descr'])
            # plt.savefig(f"{i}.png")
            # plt.close()

    raw_trajs = glob.glob(os.path.join(RAW_PREFIX, category, domain, task, "*", "*", "*", "*", "*"))
    for traj in raw_trajs:
        with open(os.path.join(traj, "obs_dict.pkl"), "rb") as f:
            obs_dict = pickle.load(f)
        labels = timestamp_to_lang[obs_dict['time_stamp'][0]]
        with open(os.path.join(traj, "lang.txt"), "w+") as f:
            for l in labels:
                f.write(l)
                f.write('\n')


    


