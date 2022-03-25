from base.utils import load_pickle
from base.utils import ensure_dir
import os
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt



raw_prediction_path = "H:\\prediction"
emotion_list = ["valence", "arousal"]
model_list = ["0", "1", "2", "3", "4", "5"]

combined_prediction_path = os.path.join(raw_prediction_path, "combined")
combined_pred_list = []

### Step 1, Generate the combined prediction to be fed to Muse-Toolkit
# Read the test set list
testset_list = []
fps_list = []
length_list = []
dataset_info = load_pickle("F:\\affwild2_abaw3_VA_processed\\dataset_info.pkl")
processing_records = load_pickle("F:\\affwild2_abaw3_VA_processed\\processing_records.pkl")

for idx, trial in processing_records.items():
    if trial['VA']['partition'] == "extra":
        testset_list.append(trial['trial'])
        fps_list.append(trial['fps'])
        length_list.append(trial['length'])


for emotion in emotion_list:

    for idx, file in enumerate(testset_list):

        save_path = os.path.join(combined_prediction_path, emotion, file + ".csv")
        combined_pred_list.append(save_path)
        if not os.path.isfile(save_path):
            ensure_dir(save_path)

            fps = fps_list[idx]
            length = length_list[idx]
            time_stamp = np.arange(length) / fps
            prediction_list = []

            for model in model_list:
                path = os.path.join(raw_prediction_path, emotion, model, file + ".txt")
                pred = pd.read_csv(path, header=None, skiprows=1).values[:, 0]
                prediction_list.append(pred)

            prediction = np.c_[time_stamp, np.stack(prediction_list).T]
            col = ["timeStamp", "0", "1", "2", "3", "4", "5"]
            df = pd.DataFrame(data=prediction, index=None, columns=col)
            df.to_csv(save_path, sep=",", index=False)

### Step 2, Carry out [fuse, alignment] on the predictions.
# https://github.com/lstappen/MuSe-Toolbox
muse_toolbox_path = "E:\\ABAW3\\muse-toolbox"
methods = [["cccc", "none"]]

for fuse, align in methods:
    for emotion in emotion_list:
        output = "output/" + fuse + "_" + align
        command = "python {} gold_standard -inp {} " \
                  "-out {} --fusion {}  --alignment {} -dim {} " \
                  "--pre_smoothing savgol --pre_smoothing_window 5 --post_smoothing_window 15 --annotators 0 1 2 3 4 5".format(
            muse_toolbox_path, combined_prediction_path, output, fuse, align, emotion)
        subprocess.call(command)

#### Step 3, Clip the prediction to [-1, +1], combine the two emotions, remove the time stamp column, and save to txt files.
#### Need to manually copy all the previously fused (and aligned if any) prediction csv files to the prediction_path.
#### The file structure in prediction_path should look like:
#### vanilla0
####    valence (contain 152 txt files)
####    arousal (contain 152 txt files)
#### vanilla1
####    valence (contain 152 txt files)
####    arousal (contain 152 txt files)
#### cccc
####    valence (contain 152 txt files)
####    arousal (contain 152 txt files)
#### ....
#### Vanilla0 means fold0 :)

prediction_path = "E:\\output_without_smoothing"
final_output_path = "E:\\ABAW3_Submission2"
# methods = ["vanilla", "mean", "cccc", "ewe", "baaw"]
methods = ["vanilla0", "vanilla1", "vanilla2", "vanilla3", "vanilla4", "vanilla5", "cccc"]
columns = ["valence", "arousal"]
for method in methods:
    for file in testset_list:
        print("Processing {} - {}".format(method, file))
        data_list = []

        output_path = os.path.join(final_output_path, method, file + ".txt")
        ensure_dir(output_path)

        if os.path.isfile(output_path):
            continue

        for emotion in emotion_list:
            path = os.path.join(prediction_path, method, emotion, file)

            if os.path.isfile(path + ".txt"):
                filename = path + ".txt"
            elif os.path.isfile(path + ".csv"):
                filename = path + ".csv"
            else:
                raise ValueError("Missing {}!".format(file))

            data = pd.read_csv(filename, header=None, skiprows=1)
            if "vanilla" not in method:
                data = data.values[:, 1][:, None]
            data_list.append(data)
        combined = np.hstack(data_list)
        combined_clipped = np.clip(combined, a_min=-1, a_max=1)

        df = pd.DataFrame(data=combined_clipped, columns=columns, index=None)
        df.to_csv(output_path, sep=",", index=False)


# #### Step 4. Visualize
#
# colors = ["tab:blue", "tab:brown", "tab:orange", "tab:pink", "tab:green", "tab:gray", "tab:red", "tab:olive", "tab:purple", "tab:cyan"]
#
# methods = ["vanilla0", "vanilla1", "vanilla2", "vanilla3", "vanilla4", "vanilla5", "cccc"]
# # methods = ["cccc", "ewe", "mean"]
# for file in testset_list:
#     fig, axs = plt.subplots(2, 1, figsize=(9, 6))
#     axs[0].set_ylim([-1, 1])
#     axs[1].set_ylim([-1, 1])
#     for idx, method in enumerate(methods):
#         path = os.path.join(final_output_path, method, file + ".txt")
#         data = pd.read_csv(path, skiprows=1, header=None).values
#
#         label = method
#         c = colors[idx]
#         for i in range(2):
#             array = data[:, i]
#             axs[i].plot(np.arange(len(array)), array, c=c, label=label)
#
#         axs[0].legend(shadow=True, fancybox=True, loc="lower left", bbox_to_anchor=[0, 1], ncol=3,)
#         axs[1].legend(shadow=True, fancybox=True, loc="lower left", bbox_to_anchor=[0, 1], ncol=3,)
#     plt.show()
#     print(0)





