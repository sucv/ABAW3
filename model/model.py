from model.arcface_model import Backbone
from model.temporal_convolutional_model import TemporalConvNet
from model.transformer import MultimodalTransformerEncoder

import os
import torch
from torch import nn


from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class my_res50(nn.Module):
    def __init__(self, input_channels=3, num_classes=8, use_pretrained=True, state_dict_name='', root_dir='', mode="ir",
                 embedding_dim=512):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, num_layers=50, drop_ratio=0.4, mode=mode)
        if use_pretrained:
            path = os.path.join(root_dir, state_dict_name + ".pth")
            state_dict = torch.load(path, map_location='cpu')

            if "backbone" in list(state_dict.keys())[0]:

                self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                        Dropout(0.4),
                                                        Flatten(),
                                                        Linear(embedding_dim * 5 * 5, embedding_dim),
                                                        BatchNorm1d(embedding_dim))

                new_state_dict = {}
                for key, value in state_dict.items():

                    if "logits" not in key:
                        new_key = key[9:]
                        new_state_dict[new_key] = value

                self.backbone.load_state_dict(new_state_dict)
            else:
                self.backbone.load_state_dict(state_dict)

            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                Dropout(0.4),
                                                Flatten(),
                                                Linear(embedding_dim * 5 * 5, embedding_dim),
                                                BatchNorm1d(embedding_dim))

        self.logits = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x, extract_cnn=False):
        x = self.backbone(x)

        if extract_cnn:
            return x

        x = self.logits(x)
        return x


class LeaderFollowerAttentionNetworkWithMultiHead(nn.Module):
    def __init__(self, backbone_state_dict, modality=['frame'], kernel_size=5, example_length=300, tcn_attention=0,
                 tcn_channel={'video': [512, 256, 256, 128], 'cnn_res50': [512, 256, 256, 128], 'mfcc':[32, 32, 32, 32], 'vggish': [32, 32, 32, 32]},
                 embedding_dim={'video': 512,  'bert': 768, 'cnn_res50': 512, 'mfcc': 39, 'vggish': 128, 'egemaps': 23},
                 encoder_dim={'video': 128, 'bert': 128, 'cnn_res50': 128, 'mfcc': 32, 'vggish': 32, 'egemaps': 32},
                 modal_dim=32, num_heads=2,
                 root_dir='', device='cuda'):
        super().__init__()
        self.backbone_state_dict = backbone_state_dict
        self.root_dir = root_dir
        self.device = device
        self.modality = modality
        self.kernel_size = kernel_size
        self.example_length = example_length
        self.tcn_channel = tcn_channel
        self.tcn_attention = tcn_attention
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.outputs = {}
        self.temporal, self.fusion = nn.ModuleDict(), None
        self.num_heads = num_heads
        self.modal_dim = modal_dim
        self.final_dim = self.encoder_dim[self.modality[0]] + self.modal_dim*len(self.modality)
        self.spatial = None

    def init(self):

        spatial = my_res50(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, self.backbone_state_dict + ".pth"), map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(num_inputs=self.embedding_dim[modal],
                                                   num_channels=self.tcn_channel[modal],
                                                   kernel_size=self.kernel_size, dropout=0.1).to(self.device)

        self.fusion = MultimodalTransformerEncoder(modalities=self.modality, input_dim=self.encoder_dim,
                                                   modal_dim=self.modal_dim, num_heads=self.num_heads,
                                                   dropout=0.1)

        self.regressor = nn.Linear(self.final_dim, 1)

    def forward(self, x):

        if 'video' in x:
            batch_size, _, channel, width, height = x['video'].shape # [batch, length, 3, 40, 40]
            x['video'] = x['video'].view(-1, channel, width, height) # [batch x length, 3, 40, 40]
            x['video'] = self.spatial(x['video']) # [batch x length, 512]
            x['video'] = x['video'].view(batch_size, self.example_length, -1).transpose(1, 2).contiguous() # [batch, 512, length]
        else:
            batch_size, _, _, _ = x[self.modality[0]].shape # [batch, 1, length, input_dim]

        for modal in self.modality:
            if modal != 'video':

                if len(x[modal]) > 1: # When batch_size > 1
                    x[modal] = x[modal].squeeze().transpose(1, 2).contiguous().float()
                else: # When batch_size = 1
                    x[modal] = x[modal].squeeze()[None, :, :].transpose(1, 2).contiguous().float()

            # Three parallel TCNs
            x[modal] = self.temporal[modal](x[modal]).transpose(1, 2).contiguous() # [batch, length, temporal_feature_dim]

        # Co-attention fusion block
        follower = self.fusion(x)   # [batch, length, 96 (32 x 3)]

        x = torch.cat((x[self.modality[0]], follower), dim=-1) # [batch, length, 244 (32 x 3 + 128)]
        x = self.regressor(x) # [batch, length, 1] 

        return x

