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
        "cnn_res50": 1,
        "VA_continuous_label": 1,
        "mfcc": 1,
        "egemaps": 1,
        "vggish": 1,
        "bert": 1
    },

    "feature_dimension": {
        "video": (48, 48, 3),
        "cnn_res50": (512,),
        "VA_continuous_label": (1,),
        "mfcc": (39,),
        "egemaps": (23,),
        "vggish": (128,),
        "bert": (768,),
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
            'egemaps': [EGEMAPS_TEMPORAL_DIM, EGEMAPS_TEMPORAL_DIM, EGEMAPS_TEMPORAL_DIM, EGEMAPS_TEMPORAL_DIM],
            'bert': [BERT_TEMPORAL_DIM // 2, BERT_TEMPORAL_DIM // 2, BERT_TEMPORAL_DIM // 4, BERT_TEMPORAL_DIM // 4]
        },
        "kernel_size": 5,
        "dropout": 0.1,
        "attention": 0,
    },

    # "tcn": {
    #     "embedding_dim": VIDEO_EMBEDDING_DIM,
    #     "channels": {
    #         'video': [VIDEO_EMBEDDING_DIM // 2, VIDEO_EMBEDDING_DIM // 2, VIDEO_EMBEDDING_DIM // 4,
    #                   VIDEO_EMBEDDING_DIM // 4],
    #         'cnn_res50': [VIDEO_EMBEDDING_DIM // 2, VIDEO_EMBEDDING_DIM // 2, VIDEO_EMBEDDING_DIM // 4,
    #                       VIDEO_EMBEDDING_DIM // 4],
    #         'mfcc': [MFCC_TEMPORAL_DIM, MFCC_TEMPORAL_DIM, MFCC_TEMPORAL_DIM, MFCC_TEMPORAL_DIM],
    #         'vggish': [VGGISH_DIM // 2, VGGISH_DIM // 2, VGGISH_DIM // 4, VGGISH_DIM // 4],
    #         'egemaps': [EGEMAPS_TEMPORAL_DIM, EGEMAPS_TEMPORAL_DIM, EGEMAPS_TEMPORAL_DIM, EGEMAPS_TEMPORAL_DIM],
    #         'bert': [BERT_TEMPORAL_DIM // 2, BERT_TEMPORAL_DIM // 2, BERT_TEMPORAL_DIM // 4, BERT_TEMPORAL_DIM // 4]
    #     },
    #     "kernel_size": 5,
    #     "dropout": 0.1,
    #     "attention": 0,
    # },

    "time_delay": 0,
    "metrics": ["rmse", "pcc", "ccc"],
    "save_plot": 0,

    "backbone": {
        "state_dict": "res50_ir_0.887",
        "mode": "ir",
    },
}
