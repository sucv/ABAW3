from models.arcface_model import Backbone
from models.temporal_convolutional_model import TemporalConvNet
from models.transformer import MultimodalTransformerEncoder, IntraModalTransformerEncoder, InterModalTransformerEncoder
from models.backbone import VisualBackbone, AudioBackbone

import math
import os
import torch
from torch import nn

import numpy as np

from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module
import torch.nn.functional as F

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


class LeaderFollowerAttentionNetwork(nn.Module):
    def __init__(self, backbone_state_dict, modality=['frame'], kernel_size=5, example_length=300, tcn_attention=0,
                 tcn_channel={'video': [512, 256, 256, 128], 'cnn_res50': [512, 256, 256, 128], 'mfcc':[32, 32, 32, 32], 'vggish': [32, 32, 32, 32]},
                 embedding_dim={'video': 512,  'bert': 768, 'cnn_res50': 512, 'mfcc': 39, 'vggish': 128, 'egemaps': 23},
                 encoder_dim={'video': 128, 'bert': 128, 'cnn_res50': 128, 'mfcc': 32, 'vggish': 32, 'egemaps': 32}, root_dir='', device='cuda'):
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
        self.temporal, self.encoderQ, self.encoderK, self.encoderV = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()

        self.final_dim = self.encoder_dim[self.modality[0]] + 32*len(self.modality)
        self.spatial = None

    def init(self):
        self.output_dim = 1

        spatial = my_res50(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, self.backbone_state_dict + ".pth"), map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(num_inputs=self.embedding_dim[modal], max_length=self.example_length,
                                                   num_channels=self.tcn_channel[modal], attention=self.tcn_attention,
                                                   kernel_size=self.kernel_size, dropout=0.1).to(self.device)

            self.encoderQ[modal] = nn.Linear(self.encoder_dim[modal], 32)
            self.encoderK[modal] = nn.Linear(self.encoder_dim[modal], 32)
            self.encoderV[modal] = nn.Linear(self.encoder_dim[modal], 32)

        self.ln = nn.LayerNorm([len(self.modality), 32])
        self.regressor = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, x):

        if 'video' in x:
            batch_size, _, channel, width, height = x['video'].shape
            x['video'] = x['video'].view(-1, channel, width, height)
            x['video'] = self.spatial(x['video'])
            _, feature_dim = x['video'].shape
            x['video'] = x['video'].view(batch_size, self.example_length, feature_dim).transpose(1, 2).contiguous()
        else:
            batch_size, _, _, _ = x[self.modality[0]].shape

        for modal in self.modality:
            if modal != 'video':
                if len(x[modal]) > 1:
                    x[modal] = x[modal].squeeze().transpose(1, 2).contiguous().float()
                else:
                    x[modal] = x[modal].squeeze()[None, :, :].transpose(1, 2).contiguous().float()

            x[modal] = self.temporal[modal](x[modal]).transpose(1, 2).contiguous()
            x[modal] = x[modal].contiguous().view(batch_size * self.example_length, -1)

        Q = [self.encoderQ[modal](x[modal]) for modal in self.modality]
        K = [self.encoderK[modal](x[modal]) for modal in self.modality]
        V = [self.encoderV[modal](x[modal]) for modal in self.modality]

        Q = torch.stack(Q, dim=-2)
        K = torch.stack(K, dim=-2)
        V = torch.stack(V, dim=-2)

        QT = Q.permute(0, 2, 1)
        scores = torch.matmul(K, QT) / math.sqrt(32)
        scores = nn.functional.softmax(scores, dim=-1)

        follower = torch.matmul(scores, V)
        follower = self.ln(follower + V)
        follower = follower.view(follower.size()[0], -1)

        x = torch.cat((x[self.modality[0]], follower), dim=-1)
        x = self.regressor(x)
        x = x.view(batch_size, self.example_length, -1)

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
        self.output_dim = 1

        spatial = my_res50(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, self.backbone_state_dict + ".pth"), map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(num_inputs=self.embedding_dim[modal], max_length=self.example_length,
                                                   num_channels=self.tcn_channel[modal], attention=self.tcn_attention,
                                                   kernel_size=self.kernel_size, dropout=0.1).to(self.device)

        self.fusion = MultimodalTransformerEncoder(modalities=self.modality, input_dim=self.encoder_dim,
                                                   modal_dim=self.modal_dim, num_heads=self.num_heads,
                                                   dropout=0.1)

        self.regressor = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, x):

        if 'video' in x:
            batch_size, _, channel, width, height = x['video'].shape
            x['video'] = x['video'].view(-1, channel, width, height)
            x['video'] = self.spatial(x['video'])
            _, feature_dim = x['video'].shape
            x['video'] = x['video'].view(batch_size, self.example_length, feature_dim).transpose(1, 2).contiguous()
        else:
            batch_size, _, _, _ = x[self.modality[0]].shape

        for modal in self.modality:
            if modal != 'video':
                if len(x[modal]) > 1:
                    x[modal] = x[modal].squeeze().transpose(1, 2).contiguous().float()
                else:
                    x[modal] = x[modal].squeeze()[None, :, :].transpose(1, 2).contiguous().float()

            x[modal] = self.temporal[modal](x[modal]).transpose(1, 2).contiguous()

        follower = self.fusion(x)

        x = torch.cat((x[self.modality[0]], follower), dim=-1)
        x = self.regressor(x)
        x = x.view(batch_size, self.example_length, -1)

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
        self.output_dim = 1

        spatial = my_res50(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, self.backbone_state_dict + ".pth"), map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(num_inputs=self.embedding_dim[modal], max_length=self.example_length,
                                                   num_channels=self.tcn_channel[modal], attention=self.tcn_attention,
                                                   kernel_size=self.kernel_size, dropout=0.1).to(self.device)

        self.fusion = MultimodalTransformerEncoder(modalities=self.modality, input_dim=self.encoder_dim,
                                                   modal_dim=self.modal_dim, num_heads=self.num_heads,
                                                   dropout=0.1)

        self.regressor = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, x):

        if 'video' in x:
            batch_size, _, channel, width, height = x['video'].shape
            x['video'] = x['video'].view(-1, channel, width, height)
            x['video'] = self.spatial(x['video'])
            _, feature_dim = x['video'].shape
            x['video'] = x['video'].view(batch_size, self.example_length, feature_dim).transpose(1, 2).contiguous()
        else:
            batch_size, _, _, _ = x[self.modality[0]].shape

        for modal in self.modality:
            if modal != 'video':
                if len(x[modal]) > 1:
                    x[modal] = x[modal].squeeze().transpose(1, 2).contiguous().float()
                else:
                    x[modal] = x[modal].squeeze()[None, :, :].transpose(1, 2).contiguous().float()

            x[modal] = self.temporal[modal](x[modal]).transpose(1, 2).contiguous()

        follower = self.fusion(x)

        x = torch.cat((x[self.modality[0]], follower), dim=-1)
        x = self.regressor(x)
        x = x.view(batch_size, self.example_length, -1)

        return x


class LFAN(nn.Module):
    def __init__(self, backbone_settings, modality=['frame'], kernel_size=5, example_length=300, tcn_attention=0,
                 tcn_channel={'video': [512, 256, 256, 128], 'cnn_res50': [512, 256, 256, 128], 'mfcc':[32, 32, 32, 32], 'vggish': [32, 32, 32, 32], 'logmel': [32, 32, 32, 32]},
                 embedding_dim={'video': 512,  'bert': 768, 'cnn_res50': 512, 'mfcc': 39, 'vggish': 128, 'logmel': 128, 'egemaps': 88},
                 encoder_dim={'video': 128, 'bert': 128, 'cnn_res50': 128, 'mfcc': 32, 'vggish': 32, 'logmel': 32, 'egemaps': 32},
                 modal_dim=32, num_heads=2,
                 root_dir='', device='cuda'):
        super().__init__()
        self.backbone_settings = backbone_settings
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
        self.spatial = nn.ModuleDict()
        self.bn = nn.ModuleDict()


    def load_visual_backbone(self, backbone_settings):

        resnet = VisualBackbone(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, backbone_settings['visual_state_dict'] + ".pth"),
                                map_location='cpu')
        resnet.load_state_dict(state_dict)

        for param in resnet.parameters():
            param.requires_grad = False

        return resnet

    def load_audio_backbone(self, backbone_settings):

        vggish = AudioBackbone()
        state_dict = torch.load(os.path.join(self.root_dir, backbone_settings['audio_state_dict'] + ".pth"),
                                map_location='cpu')
        vggish.backbone.load_state_dict(state_dict)

        for param in vggish.parameters():
            param.requires_grad = False


        return vggish

    def init(self):
        self.output_dim = 1


        if 'video' in self.modality:
            self.root_dir = self.root_dir
            self.spatial["visual"] = self.load_visual_backbone(backbone_settings=self.backbone_settings)

        if 'logmel' in self.modality:
            self.root_dir = self.root_dir
            self.spatial["audio"] = self.load_audio_backbone(backbone_settings=self.backbone_settings)

        for modal in self.modality:

            self.temporal[modal] = TemporalConvNet(num_inputs=self.embedding_dim[modal], max_length=self.example_length,
                                                   num_channels=self.tcn_channel[modal], attention=self.tcn_attention,
                                                   kernel_size=self.kernel_size, dropout=0.1).to(self.device)
            self.bn[modal] = BatchNorm1d(self.tcn_channel[modal][-1])

        self.fusion = MultimodalTransformerEncoder(modalities=self.modality, input_dim=self.encoder_dim,
                                                   modal_dim=self.modal_dim, num_heads=self.num_heads,
                                                   dropout=0.1)

        self.regressor = nn.Linear(self.final_dim, self.output_dim)

    def forward(self, X):

        if 'video' in X:
            batch_size, length, channel, width, height = X['video'].shape
            X['video'] = X['video'].view(-1, channel, width, height) # [batch x length, channel, width, height]
            X['video'] = self.spatial.visual(X['video'])
            _, feature_dim = X['video'].shape
            X['video'] = X['video'].view(batch_size, length, feature_dim).unsqueeze(1) # [batch, 1, length, feature_dim]

        if 'logmel' in X:
            batch_size, height, length, width = X['logmel'].shape
            X['logmel'] = X['logmel'].permute((0, 2, 3, 1)).contiguous()
            X['logmel'] = X['logmel'].view(-1, width, height) # [batch x length, channel, width, height]
            X['logmel'] = self.spatial.audio(X['logmel'])
            _, feature_dim = X['logmel'].shape
            X['logmel'] = X['logmel'].view(batch_size, length, feature_dim).unsqueeze(1) # [batch, 1, length, feature_dim]

        for modal in X:
            X[modal] = X[modal].squeeze(1).transpose(1, 2)
            X[modal] = self.temporal[modal](X[modal])
            X[modal] = self.bn[modal](X[modal]).transpose(1, 2)

        follower = self.fusion(X)

        X = torch.cat((X[self.modality[0]], follower), dim=-1)
        X = self.regressor(X)
        X = X.view(batch_size, self.example_length, -1)
        X = torch.tanh(X)
        return X


class AttentionFusion(nn.Module):
    """ Fuse modalities using attention. """

    def __init__(self,
                 num_feats_modality: list,
                 num_out_feats: int = 256):
        """ Instantiate attention fusion instance.

        Args:
            num_feats_modality (list): Number of features per modality.
            num_out_feats (int): Number of output features.
        """

        super(AttentionFusion, self).__init__()

        self.attn = nn.ModuleList([])
        for num_feats in num_feats_modality:
            self.attn.append(
                nn.Linear(num_feats, num_out_feats))

        self.weights = nn.Linear(num_out_feats * len(num_feats_modality), num_out_feats * len(num_feats_modality))
        self.num_features = num_out_feats * len(num_feats_modality)

    def forward(self, x: list):
        """ Forward pass

        Args:
            x (list): List of modality tensors with dimensions (BS x SeqLen x N).
        """

        proj_m = []
        for i, m in enumerate(x.values()):
            proj_m.append(self.attn[i](m.transpose(1, 2)))

        attn_weights = F.softmax(
            self.weights(torch.cat(proj_m, -1)), dim=-1)

        out_feats = attn_weights * torch.cat(proj_m, -1)

        return out_feats


class CAN(nn.Module):
    def __init__(self, modalities, tcn_settings, backbone_settings, output_dim, root_dir, device):
        super().__init__()
        self.device = device


        self.temporal = nn.ModuleDict()
        self.up_sample = nn.ModuleDict()
        self.bn = nn.ModuleDict()

        self.spatial = nn.ModuleDict()


        for modal in modalities:
            self.temporal[modal] = TemporalConvNet(num_inputs=tcn_settings[modal]['input_dim'],
                                                   num_channels=tcn_settings[modal]['channel'],
                                                   kernel_size=tcn_settings[modal]['kernel_size'])
            self.bn[modal] = BatchNorm1d(tcn_settings[modal]['channel'][-1] )


        feas_modalities = [tcn_settings[modal]['channel'][-1] for modal in modalities]
        self.fuse = AttentionFusion(num_feats_modality=feas_modalities, num_out_feats=128)

        self.conv_c = nn.Conv1d(128 * len(modalities), 128, 1)

        self.bn1 = BatchNorm1d(128 * len(modalities))
        self.fc1 = Linear(128* len(modalities), 128* len(modalities))
        self.fc2 = Linear(128* len(modalities), output_dim)


        if 'video' in modalities:
            self.root_dir = root_dir
            self.spatial["visual"] = self.load_visual_backbone(backbone_settings=backbone_settings)

        if 'logmel' in modalities:
            self.root_dir = root_dir
            self.spatial["audio"] = self.load_audio_backbone(backbone_settings=backbone_settings)

    def load_visual_backbone(self, backbone_settings):

        resnet = VisualBackbone(mode='ir', use_pretrained=False)
        state_dict = torch.load(os.path.join(self.root_dir, backbone_settings['visual_state_dict'] + ".pth"),
                                map_location='cpu')
        resnet.load_state_dict(state_dict)

        for param in resnet.parameters():
            param.requires_grad = False


        return resnet

    def load_audio_backbone(self, backbone_settings):

        vggish = AudioBackbone()
        state_dict = torch.load(os.path.join(self.root_dir, backbone_settings['audio_state_dict'] + ".pth"),
                                map_location='cpu')
        vggish.backbone.load_state_dict(state_dict)

        for param in vggish.parameters():
            param.requires_grad = False


        return vggish


    def forward(self, X):

        x = {}

        if 'video' in X:
            batch_size, length, channel, width, height = X['video'].shape
            X['video'] = X['video'].view(-1, channel, width, height) # [batch x length, channel, width, height]
            X['video'] = self.spatial.visual(X['video'])
            _, feature_dim = X['video'].shape
            X['video'] = X['video'].view(batch_size, length, feature_dim).unsqueeze(1) # [batch, 1, length, feature_dim]

        if 'logmel' in X:
            batch_size, height, length, width = X['logmel'].shape
            X['logmel'] = X['logmel'].permute((0, 2, 3, 1)).contiguous()
            X['logmel'] = X['logmel'].view(-1, width, height) # [batch x length, channel, width, height]
            X['logmel'] = self.spatial.audio(X['logmel'])
            _, feature_dim = X['logmel'].shape
            X['logmel'] = X['logmel'].view(batch_size, length, feature_dim).unsqueeze(1) # [batch, 1, length, feature_dim]

        for modal in X:
            x[modal] = X[modal].squeeze(1).transpose(1, 2)
            x[modal] = self.temporal[modal](x[modal])
            x[modal] = self.bn[modal](x[modal])

        c = self.fuse(x)
        c = self.fc1(c).transpose(1, 2)
        c = self.bn1(c).transpose(1, 2)
        c = F.leaky_relu(c)
        c = self.fc2(c)
        c = torch.tanh(c)

        return c
