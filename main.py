import sys
import argparse

if __name__ == '__main__':
    frame_size = 48
    crop_size = 40

    parser = argparse.ArgumentParser(description='Say hello')

    # 1. Experiment Setting
    # 1.1. Server
    parser.add_argument('-gpu', default=2, type=int, help='Which gpu to use?')
    parser.add_argument('-cpu', default=1, type=int, help='How many threads are allowed?')
    parser.add_argument('-high_performance_cluster', default=0, type=int, help='On high-performance server or not?'
                                                                               'If set to 1, then the gpu and cpu settings will be ignored.'
                                                                               'It should be set to 1 if the user has no right to specify cpu and gpu usage, '
                                                                               'e.g., on Google Colab or NSCC.')

    # 1.2. Paths
    parser.add_argument('-dataset_path', default='/home/zhangsu/dataset/abaw5', type=str,
                        help='The root directory of the preprocessed dataset.')  # /scratch/users/ntu/su012/dataset/mahnob
    parser.add_argument('-load_path', default='/home/zhangsu/load', type=str,
                        help='The path to load the trained models, such as the backbone.')  # /scratch/users/ntu/su012/pretrained_model
    parser.add_argument('-save_path', default='/home/zhangsu/save', type=str,
                        help='The path to save the trained models ')  # /scratch/users/ntu/su012/trained_model
    parser.add_argument('-python_package_path', default='/home/zhangsu/ABAW5', type=str,
                        help='The path to the entire repository.')

    # 1.3. Experiment name, and stamp, will be used to name the output files.
    # Stamp is used to add a string to the outpout filename, so that instances with different setting will not overwride.
    parser.add_argument('-experiment_name', default="ABAW5", help='The experiment name.')
    parser.add_argument('-stamp', default='0206_vlb_100ep_bs12', type=str, help='To indicate different experiment instances')

    # 1.4. Load checkpoint or not?
    parser.add_argument('-resume', default=0, type=int, help='Resume from checkpoint?')

    # 1.5. Debug or not?
    parser.add_argument('-debug', default=0, type=int, help='The number of trials to load for debugging. Set to 0 for non-debugging execution.')

    # 1.6. What modality to use?
    #  Set to ['frame'] for unimodal and ['frame', 'mfcc', 'vggish' for multimodal. Using other features may cause bugs.
    parser.add_argument('-modality', default=['video', 'logmel', "egemaps", "VA_continuous_label"], nargs="*")
    # Calculate mean and std for each modality?
    parser.add_argument('-calc_mean_std', default=0,  type=int,
                        help='Calculate the mean and std and save to a pickle file')

    # 1.7. What emotion to train?
    # If choose both, then the multi-headed will be automatically enabled, meaning, the models will predict both the Valence
    #   and Arousal.
    # If choose valence or arousal, the output dimension can be 1 for single-headed, or 2 for multi-headed.
    # For the latter, a weight will be applied to the output to favor the selected emotion.
    parser.add_argument('-emotion', default="valence",
                        help='The emotion dimension to focus when updating gradient: arousal, valence, both')

    # 1.8. Whether to save the models?
    parser.add_argument('-save_model', default=1, type=int, help='Whether to save the models?')

    # 2. Training settings.
    parser.add_argument('-num_heads', default=2, type=int)
    parser.add_argument('-modal_dim', default=32, type=int)
    parser.add_argument('-tcn_kernel_size', default=5, type=int,
                        help='The size of the 1D kernel for temporal convolutional networks.')

    # 2.1. Overall settings
    parser.add_argument('-model_name', default="LFAN", help='LFAN, CAN')
    parser.add_argument('-cross_validation', default=1, type=int)
    parser.add_argument('-num_folds', default=6, type=int)
    parser.add_argument('-folds_to_run', default=[1], nargs="+", type=int, help='Which fold(s) to run? Each fold may take 1-2 days.')

    # 2.2. Epochs and data
    parser.add_argument('-num_epochs', default=100, type=int, help='The total of epochs to run during training.')
    parser.add_argument('-min_num_epochs', default=5, type=int, help='The minimum epoch to run at least.')
    parser.add_argument('-early_stopping', default=50, type=int,
                        help='If no improvement, the number of epoch to run before halting the training')
    parser.add_argument('-window_length', default=300, type=int, help='The length in point number to windowing the data.')
    parser.add_argument('-hop_length', default=200, type=int, help='The step size or stride to move the window.')
    parser.add_argument('-batch_size', default=12, type=int)

    # 2.1. Scheduler and Parameter Control
    parser.add_argument('-seed', default=3407, type=int)
    parser.add_argument('-scheduler', default='plateau', type=str, help='plateau, cosine')
    parser.add_argument('-learning_rate', default=1e-5, type=float, help='The initial learning rate.')
    parser.add_argument('-min_learning_rate', default=1.e-8, type=float, help='The minimum learning rate.')
    parser.add_argument('-patience', default=5, type=int, help='Patience for learning rate changes.')
    parser.add_argument('-factor', default=0.1, type=float, help='The multiplier to decrease the learning rate.')
    parser.add_argument('-gradual_release', default=1, type=int, help='Whether to gradually release some layers?')
    parser.add_argument('-release_count', default=3, type=int, help='How many layer groups to release?')
    parser.add_argument('-milestone', default=[0], nargs="+", type=int, help='The specific epochs to do something.')
    parser.add_argument('-load_best_at_each_epoch', default=1, type=int,
                        help='Whether to load the best models state at the end of each epoch?')

    # 2.2. Groundtruth settings
    parser.add_argument('-time_delay', default=0, type=float,
                        help='For time_delay=n, it means the n-th label points will be taken as the 1st, and the following ones will be shifted accordingly.'
                             'The rear point will be duplicated to meet the original length.'
                             'This is used to compensate the human labeling delay.')
    parser.add_argument('-metrics', default=["rmse", "pcc", "ccc"], nargs="*", help='The evaluation metrics.')
    parser.add_argument('-save_plot', default=0, type=int,
                        help='Whether to plot the session-wise output/target or not?')

    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)

    from experiment import Experiment

    exp = Experiment(args)
    exp.prepare()
    exp.run()