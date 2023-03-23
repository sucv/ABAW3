from base.experiment import GenericExperiment
from base.utils import load_pickle
from base.loss_function import CCCLoss
from trainer import Trainer

from dataset import DataArranger, Dataset
from base.checkpointer import Checkpointer
from models.model import LFAN, CAN

from base.parameter_control import ResnetParamControl

import os


class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.release_count = args.release_count
        self.gradual_release = args.gradual_release
        self.milestone = args.milestone
        self.backbone_mode = "ir"
        self.min_num_epochs = args.min_num_epochs
        self.num_epochs = args.num_epochs
        self.early_stopping = args.early_stopping
        self.load_best_at_each_epoch = args.load_best_at_each_epoch

        self.num_heads = args.num_heads
        self.modal_dim = args.modal_dim
        self.tcn_kernel_size = args.tcn_kernel_size

    def prepare(self):
        self.config = self.get_config()

        self.feature_dimension = self.get_feature_dimension(self.config)
        self.multiplier = self.get_multiplier(self.config)
        self.time_delay = self.get_time_delay(self.config)

        self.get_modality()
        self.continuous_label_dim = self.get_selected_continuous_label_dim()

        self.dataset_info = load_pickle(os.path.join(self.dataset_path, "dataset_info.pkl"))
        self.data_arranger = self.init_data_arranger()
        if self.calc_mean_std:
            self.calc_mean_std_fn()
        self.mean_std_dict = load_pickle(os.path.join(self.dataset_path, "mean_std_info.pkl"))

    def init_data_arranger(self):
        arranger = DataArranger(self.dataset_info, self.dataset_path, self.debug)
        return arranger

    def run(self):

        criterion = CCCLoss()

        for fold in iter(self.folds_to_run):

            save_path = os.path.join(self.save_path,
                                     self.experiment_name + "_" + self.model_name + "_" + self.stamp + "_fold" + str(
                                         fold) + "_" + self.emotion +  "_seed" + str(self.seed))
            os.makedirs(save_path, exist_ok=True)

            checkpoint_filename = os.path.join(save_path, "checkpoint.pkl")

            model = self.init_model()

            dataloaders = self.init_dataloader(fold)

            trainer_kwards = {'device': self.device, 'emotion': self.emotion, 'model_name': self.model_name,
                              'models': model, 'save_path': save_path, 'fold': fold,
                              'min_epoch': self.min_num_epochs, 'max_epoch': self.num_epochs,
                              'early_stopping': self.early_stopping, 'scheduler': self.scheduler,
                              'learning_rate': self.learning_rate, 'min_learning_rate': self.min_learning_rate,
                              'patience': self.patience, 'batch_size': self.batch_size,
                              'criterion': criterion, 'factor': self.factor, 'verbose': True,
                              'milestone': self.milestone, 'metrics': self.config['metrics'],
                              'load_best_at_each_epoch': self.load_best_at_each_epoch,
                              'save_plot': self.config['save_plot']}

            trainer = Trainer(**trainer_kwards)

            parameter_controller = ResnetParamControl(trainer, gradual_release=self.gradual_release,
                                                      release_count=self.release_count,
                                                      backbone_mode=["visual", "audio"])

            checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger(self.args, self.config)

            if not trainer.fit_finished:
                trainer.fit(dataloaders, parameter_controller=parameter_controller,
                            checkpoint_controller=checkpoint_controller)

            test_kwargs = {'dataloader_dict': dataloaders, 'epoch': None, 'partition': 'extra'}
            trainer.test(checkpoint_controller, predict_only=1, **test_kwargs)

    def init_dataset(self, data, continuous_label_dim, mode, fold):
        dataset = Dataset(data, continuous_label_dim, self.modality, self.multiplier,
                          self.feature_dimension, self.window_length,
                          mode, mean_std=self.mean_std_dict[fold][mode], time_delay=self.time_delay)
        return dataset

    def init_model(self):
        self.init_randomness()
        modality = [modal for modal in self.modality if "continuous_label" not in modal]

        if self.model_name == "LFAN":
            model = LFAN(backbone_settings=self.config['backbone_settings'],
                                                   modality=modality, example_length=self.window_length,
                                                   kernel_size=self.tcn_kernel_size,
                                                   tcn_channel=self.config['tcn']['channels'], modal_dim=self.modal_dim, num_heads=self.num_heads,
                                                   root_dir=self.load_path, device=self.device)
            model.init()
        elif self.model_name == "CAN":
            model = CAN(root_dir=self.load_path, modalities=modality, tcn_settings=self.config['tcn_settings'], backbone_settings=self.config['backbone_settings'], output_dim=len(self.continuous_label_dim), device=self.device)

        return model

    def get_modality(self):
        pass

    def get_config(self):
        from configs import config
        return config

    def get_selected_continuous_label_dim(self):
        if self.emotion == "arousal":
            dim = [1]
        elif self.emotion == "valence":
            dim = [0]
        else:
            raise ValueError("Unknown emotion!")
        return dim
