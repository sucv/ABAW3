> During my PhD I struggled for finding an end-to-end working pipeline for my emotion recognition project. I was new to deep learning, I didn't have seniors to follow, and the ER community is not as popular/open as others. Then a good soul shared his code and models with me. I was salvaged and survived my PhD.
> I hope the code and model state here can be helpful for those lost souls.

### Conda environment

```
conda create --name abaw2 pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install tqdm matplotlib scipy pandas
```

### Code for preprocessing

[url](https://drive.google.com/file/d/1_5HkqdQrf388JJvLAH1B_d7ctZLWy1KE/view?usp=drive_link)

### Model state dict

[url](https://drive.google.com/drive/folders/1yh0vVY-AlwVCDxdAUy_c6e9fVbXUSnF5?usp=sharing)

### Specify the settings

In main.py:

- Adjust the four paths in 1.2 for your machine.
- In 1.3, name your experiment by specifying `-stamp` argument. When you are going to run the `main.py`, you need to carefully name your instance. Name determines the output directory. If two instances have the same name, then the late one will replace the early one and ruin its output.
- In 1.4, to resume an instance, add -resume 1 to the command. For example, `python main.py -resume 0` will start a fresh run, and `python main.py -resume 1` with continue an existing instance from the checkpoint.
- In 1.5, to efficiently debug, specify `-debug 1` so that only one trial will be loaded for each fold.
- In 1.7, specify `-emotion` to either `arousal` or `valence`.
- In 2.1, specify `-folds_to_run` to 0-6. For example, `-folds_to_run 0` runs fold 0. `-folds_to_run 0 1 2` runs fold 0, fold 1, and fold 2 in a row.

### Run the code

Usually, with all the default settings in `main.py` being correctly set, all u need to type in is like below.

```
python main.py -folds_to_run 0 -emotion "valence" -stamp "cv"
```


Of course, if you have more machines available, u can run one fold on each machine.


Note that one single fold may take 1-2 days. So the following command may take 5 days to complete:

```
python main.py -folds_to_run 0 1 2 -emotion "valence" -stamp "cv"
```

Sometimes, the running is stopped falsely. To continue with the latest epoch, add `-resume 1` to the last command you were running like:

```
python main.py -folds_to_run 0 -emotion "valence" -stamp "cv" -resume 1
```

### Collect the result

The results will be saved in your specified `-save_path`, which include:

- training logs in csv format;
- trained model state dict and the checkpoint.
- predictions on unseen partition.






