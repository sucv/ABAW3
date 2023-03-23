import pickle

file1 = "/home/zhangsu/save_path/ABAW3_LFA_restore_fold0_valence/checkpoint.pkl"
file2 = "/home/zhangsu/save_path/ABAW3_LFA_VAL_fold0_valence/checkpoint.pkl"

with open(file1, "rb") as input_file:
    e = pickle.load(input_file)

with open(file2, "rb") as input_file:
    f = pickle.load(input_file)

print(0)