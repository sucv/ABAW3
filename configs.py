VIDEO_EMBEDDING_DIM = 512
MFCC_DIM = 39
VGGISH_DIM = 128
EGEMAPS_DIM = 23
BERT_DIM = 768
VIDEO_TEMPORAL_DIM = 128
MFCC_TEMPORAL_DIM = 32
VGGISH_TEMPORAL_DIM = 32
EGEMAPS_TEMPORAL_DIM = 32
BERT_TEMPORAL_DIM = 512

config = {
    "frequency": {
        "video": None,
        "continuous_label": None,
        "mfcc": 100,
        "egemaps": 100,
        "vggish": None,
        "bert": None
    },

    "multiplier": {
        "video": 1,
        "cnn": 1,
        "AU_continuous_label": 1,
        "EXPR_continuous_label": 1,
        "VA_continuous_label": 1,
        "continuous_label": 1,
        "mfcc": 1,
        "egemaps": 1,
        "vggish": 1,
        "logmel": 1,
        "bert": 1,
    },

    "feature_dimension": {
        "video": (48, 48, 3),
        "cnn": (512,),
        "AU_continuous_label": (12,),
        "EXPR_continuous_label": (1,),
        "VA_continuous_label": (1,),
        "continuous_label": (1,),
        "SSL_continuous_label": (4,),
        "mfcc": (39,),
        "egemaps": (88,),
        "vggish": (128,),
        "logmel": (96, 64),
        "bert": (768,)
    },

    "tcn": {
        "embedding_dim": VIDEO_EMBEDDING_DIM,
        "channels": {
            'video': [VIDEO_EMBEDDING_DIM // 2, VIDEO_EMBEDDING_DIM // 2, VIDEO_EMBEDDING_DIM // 4,
                      VIDEO_EMBEDDING_DIM // 4],
            'cnn_res50': [VIDEO_EMBEDDING_DIM // 2, VIDEO_EMBEDDING_DIM // 2, VIDEO_EMBEDDING_DIM // 4,
                          VIDEO_EMBEDDING_DIM // 4],
            'mfcc': [MFCC_TEMPORAL_DIM, MFCC_TEMPORAL_DIM, MFCC_TEMPORAL_DIM, MFCC_TEMPORAL_DIM],
            'vggish': [VGGISH_DIM // 2, VGGISH_DIM // 2, VGGISH_DIM // 4, VGGISH_DIM // 4],
            'logmel': [VGGISH_DIM // 2, VGGISH_DIM // 2, VGGISH_DIM // 4, VGGISH_DIM // 4],
            'egemaps': [EGEMAPS_TEMPORAL_DIM, EGEMAPS_TEMPORAL_DIM, EGEMAPS_TEMPORAL_DIM, EGEMAPS_TEMPORAL_DIM],
            'bert': [BERT_TEMPORAL_DIM // 2, BERT_TEMPORAL_DIM // 2, BERT_TEMPORAL_DIM // 4, BERT_TEMPORAL_DIM // 4]
        },
        "kernel_size": 5,
        "dropout": 0.1,
        "attention": 0,
    },

    "tcn_settings": {
        "video": {
            "input_dim": 512,
            "channel": [256, 256, 128, 128, 128],
            "kernel_size": 5
        },
        "cnn": {
            "input_dim": 512,
            "channel": [256, 256, 128, 128],
            "kernel_size": 5
        },
        "cnn_res50": {
            "input_dim": 512,
            "channel": [256, 256, 128, 128],
            "kernel_size": 5
        },
        "vggish": {
            "input_dim": 128,
            "channel": [128, 128, 64, 64],
            "kernel_size": 5
        },

        "logmel": {
            "input_dim": 128,
            "channel": [128, 128, 64, 64, 64],
            "kernel_size": 5
        },

        "egemaps": {
            "input_dim": 88,
            "channel": [64, 64, 32, 32],
            "kernel_size": 5
        },
        "mfcc": {
            "input_dim": 39,
            "channel": [32, 32, 32, 32],
            "kernel_size": 5
        },
        "landmark": {
            "input_dim": 136,
            "channel": [64, 64, 32, 32],
            "kernel_size": 5
        },
        "bert": {
            "input_dim": 768,
            "channel": [256, 256, 128, 128],
            "kernel_size": 5
        }
    },

    "vae_settings": {
        "input_dim": 128
    },

    "attn_settings": {
        "input_dim": 128,
        "embedding_dim": 64,
        "num_head": 4
    },

    "backbone_settings": {
        "visual_state_dict": "res50_ir_0.887",
        "audio_state_dict": "vggish"
    },


    "time_delay": 0,
    "metrics": ["rmse", "pcc", "ccc"],
    "save_plot": 0,

    "backbone": {
        "state_dict": "res50_ir_0.887",
        "mode": "ir",
    },
}
