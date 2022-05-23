### Please cite [our paper](https://arxiv.org/pdf/2203.13031.pdf) 

```
@article{zhang2022continuous,
  title={Continuous Emotion Recognition using Visual-audio-linguistic information: A Technical Report for ABAW3},
  author={Zhang, Su and An, Ruyi and Ding, Yi and Guan, Cuntai},
  journal={arXiv preprint arXiv:2203.13031},
  year={2022}}
```

### Conda environment

```
conda create --name abaw2 pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install tqdm matplotlib scipy pandas
```

### Download data and backbone state dict

Only the backbone state dict is available. As for the data, please access via the ABAW organizer.


### Specify the settings

In `main.py`:

- Adjust the four paths in 1.2 for your machine.
    - `-data_path` should contain `compacted_48`, `dataset_info.pkl` and `mean_std_info.pkl`.
        - `mean_std_info.pkl` keeps the mean and std for all the features on the 6 folds. You may need to calculate your own `mean_std_info.pkl` as the data partitioning may be different across systems and environments. See 1.6 below.
    - `-load_path` should contain the backbone state dict.
    - `-save_path` can be anywhere appropriately in your machine.
    - `-python_package_path` should be `/path/to/the/code/ABAW3`.
- In 1.3, name your experiment by specifying `-stamp` argument. When you are going to run the `main.py`, you need to carefully name your instance. Name determines the output directory. If two instances have the same name, then the late one will replace the early one and ruin its output.
- In 1.4, to resume an instance, add -resume 1 to the command. For example, `python main.py -resume 0` will start a fresh run, and `python main.py -resume 1` with continue an existing instance from the checkpoint.
- In 1.5, to efficiently debug, specify `-debug 1` so that only one trial will be loaded for each fold.
- In 1.6, in practice, if you are sure that the data would be split to the same exact 6 folds, then set it to 1 for only the first time, and set it to 0 for further run. So that the further training can skip the time-consuming calculation (up to 20-40 mins).
- In 1.7, specify `-emotion` to either `arousal` or `valence`.
- In 2.1, specify `-folds_to_run` to 0-5. For example, `-folds_to_run 0` runs fold 0. `-folds_to_run 0 1 2` runs fold 0, fold 1, and fold 2 in a row.

In `configs.py`:

- Specify `state_dict` of `backbone` as the backbone filename.


### Run the code

Usually, with all the default settings in `main.py` being correctly set, all u need to type in is like below.

```
python main.py -folds_to_run 0 -emotion "valence" -stamp "cv"
```


Of course, if you have more machines available, u can run one fold on each machine.


Note that one single fold may take 2-3 days. So the following command may take more than 1 week to complete:

```
python main.py -folds_to_run 0 1 2 -emotion "valence" -stamp "cv"
```

Therefore we strongly recommend to specify `-folds_to_run` to only one fold. 

Sometimes, the training may be stopped unexpectedly. To continue with the latest epoch, add `-resume 1` to the last command you were running like:

```
python main.py -folds_to_run 0 -emotion "valence" -stamp "cv" -resume 1
```

### Collect the result

The results will be saved in your specified `-save_path`, which include:

- training logs in csv format;
- visualization in jpg format;
- trained model state dict and the checkpoint;
- predictions on unseen partition.











