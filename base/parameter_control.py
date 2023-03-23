from operator import itemgetter

import numpy as np


class GenericParamControl(object):
    @staticmethod
    def init_module_list():
        raise NotImplementedError

    @staticmethod
    def init_param_group():
        raise NotImplementedError

    def get_param_group(self):
        raise NotImplementedError

    def release_param(self, model):
        raise NotImplementedError


class ResnetParamControl(GenericParamControl):
    r""""
    It controls what layers to release.
    """

    def __init__(self, trainer, gradual_release=1, release_count=8, backbone_mode="ir"):
        # It effects the models and optimizer of trainer.
        self.trainer = trainer

        # 0 to disable the gradual release.
        self.gradual_release = gradual_release

        # How many times to release.
        self.release_count = release_count

        # The backbone mode of the Resnet, usually ir or ir_se.
        self.backbone_mode = backbone_mode

        # The layer indices to be grouped. One group will be released at a time.
        self.module_dict = self.init_module_list()

        # Obtain the parameters according to the indices.
        self.module_stack = self.init_param_group()

        # A flag indicating whether to halt the trainer.
        self.early_stop = False

    def init_module_list(self):

        # 4-10 are the output layers. 163-187 are the fourth block of Resnet 50. 142-163 are half of the third block of Resnet 50.
        # module_list = [[(4, 10), (163, 187)], [(142, 163)]]
        module_dict = {"visual": [[(4, 10)], [(163, 187)], [(142, 163)]], "audio": [[(16, 18)], [(14, 16)], [(12, 14)]]}
        # module_dict = {"visual": [[(4, 10), (173, 187), (156, 173)]], "audio": [[(16, 18), (14, 16), (12, 14)]]}
        return module_dict


    def init_param_group(self):
        # Get the parameters according to the module list.
        module_stack = {"visual": [], "audio": []}
        for modal, ranges in self.module_dict.items():

            for groups in ranges:
                slice_range = []
                for group in groups:
                    slice_range += (list(np.arange(*group)))

                module_stack[modal].append(slice_range)
        return module_stack

    def get_param_group(self, modal):
        # Pop out the last element of the stack.
        modules_to_release = self.module_stack[modal].pop(0)
        return modules_to_release

    def get_current_lr(self):
        current_lr = self.trainer.optimizer.param_groups[0]['lr']
        return current_lr

    def release_param(self, model, epoch=0, modalities=['visual', "audio"]):
        if self.gradual_release:
            if self.release_count > 0:
                for modal in modalities:

                    if modal not in model:
                        continue

                    indices = self.get_param_group(modal)

                    for param in list(itemgetter(*indices)(list(model[modal].parameters()))):
                        param.requires_grad = True

                self.trainer.init_optimizer_and_scheduler(epoch=epoch)
                self.release_count -= 1
                self.trainer.early_stopping_counter = self.trainer.early_stopping
            else:
                print("Early stopped since no further parameters to release!")
                self.early_stop = True

    def load_trainer(self, trainer):

        # If started from a checkpoint, then load the trainer.
        self.trainer = trainer


