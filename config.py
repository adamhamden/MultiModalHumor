import torch


def config():
    batch_size = 32
    learning_rate = .01
    epochs = 5
    context_length = 64
    sequence_length = 64 / context_length

    use_text = True
    use_audio = False
    use_pose = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lstm_text_input = 768
    lstm_audio_input = 128
    lstm_pose_input = 104
    lstm_text_hidden_size = 128
    lstm_audio_hidden_size = 32
    lstm_pose_hidden_size = 32

    mmcn_text_dropout = .1
    mmcn_audio_dropout = .1
    mmcn_pose_dropout = .1
    mmcn_text_input = context_length * lstm_text_hidden_size
    mmcn_audio_input = context_length * lstm_audio_hidden_size
    mmcn_pose_input = context_length * lstm_pose_hidden_size
    mmcn_text_hidden_size = 128
    mmcn_audio_hidden_size = 32
    mmcn_pose_hidden_size = 32
    mmcn_dropout = .1

    transformer_src_size = lstm_text_hidden_size + lstm_audio_hidden_size + lstm_pose_hidden_size
    transformer_tgt_size = 256
    transformer_sequence_length = context_length
    transformer_dropout = .1
    d_model = 512
    n_heads = 6
    n_layers = 8
    d_feedforward = 2048
    d_value = 64
    d_key = 64

    return locals()


humor_speakers = ['oliver',  # TV sitting high_freq
                  'jon',  # TV sitting
                  'conan',  # TV standing high_freq
                  'ellen',  # TV standing
                  'seth',  # TV sitting low frequency
                  'colbert',  # TV standing high_freq
                  'corden',  # TV standing
                  'fallon',  # TV standing
                  'huckabee',  # TV standing
                  'maher',  # TV standing
                  'minhaj',  # TV standing
                  'bee',  # TV standing
                  'noah'  # TV sitting
                  ]

speakers = ['oliver',  # TV sitting high_freq
            'jon',  # TV sitting
            'conan',  # TV standing high_freq
            'rock',  # lec sitting
            'chemistry',  # lec sitting
            'ellen',  # TV standing
            'almaram',  # eval sitting
            'angelica',  # eval sitting
            'seth',  # TV sitting low frequency
            'shelly',  # TV sitting
            'colbert',  # TV standing high_freq
            'corden',  # TV standing
            'fallon',  # TV standing
            'huckabee',  # TV standing
            'maher',  # TV standing
            'lec_cosmic',  # lec sitting
            'lec_evol',  # lec sitting
            'lec_hist',  # lec sitting
            'lec_law',  # lec sitting
            'minhaj',  # TV standing
            'ytch_charisma',  # yt sitting
            'ytch_dating',  # yt sitting
            'ytch_prof',  # yt sitting
            'bee',  # TV standing
            'noah'  # TV sitting
            ]
