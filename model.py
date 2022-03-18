import torch
import torch.nn as nn
from transformer import Transformer


class UnimodalContextNetwork(nn.Module):
    def __init__(self, config):
        super(UnimodalContextNetwork, self).__init__()
        self.config = config
        self.device = config['device']

        self.t_lstm = nn.LSTM(input_size=config['unimodal_context']['text_lstm_input'],
                              hidden_size=config['unimodal_context']['text_lstm_hidden_size'], batch_first=True)
        self.a_lstm = nn.LSTM(input_size=config['unimodal_context']['audio_lstm_input'],
                              hidden_size=config['unimodal_context']['audio_lstm_hidden_size'], batch_first=True)
        self.p_lstm = nn.LSTM(input_size=config['unimodal_context']['pose_lstm_input'],
                              hidden_size=config['unimodal_context']['pose_lstm_hidden_size'], batch_first=True)

    def forward(self, x_t, x_a, x_p):

        batch_size, context_length, seq_length, _ = x_t.shape
        new_batch_size = batch_size * context_length

        x_t.to(self.device)
        x_a.to(self.device)
        x_p.to(self.device)

        x_t = torch.reshape(x_t, (new_batch_size, seq_length, self.config['unimodal_context']['text_lstm_input']))
        x_a = torch.reshape(x_a, (new_batch_size, seq_length, self.config['unimodal_context']['audio_lstm_input']))
        x_p = torch.reshape(x_p, (new_batch_size, seq_length, self.config['unimodal_context']['pose_lstm_input']))

        if not self.config['use_text']:
            x_t = torch.zeros_like(x_t, requires_grad=True)
        if not self.config['use_audio']:
            x_a = torch.zeros_like(x_a, requires_grad=True)
        if not self.config['use_pose']:
            x_p = torch.zeros_like(x_p, requires_grad=True)

        text_h0 = torch.zeros(new_batch_size, self.config['unimodal_context']['text_lstm_hidden_size']).unsqueeze(0).to(
            self.device)
        audio_h0 = torch.zeros(new_batch_size, self.config['unimodal_context']['audio_lstm_hidden_size']).unsqueeze(
            0).to(
            self.device)
        pose_h0 = torch.zeros(new_batch_size, self.config['unimodal_context']['pose_lstm_hidden_size']).unsqueeze(0).to(
            self.device)

        text_c0 = torch.zeros(new_batch_size, self.config['unimodal_context']['text_lstm_hidden_size']).unsqueeze(0).to(
            self.device)
        audio_c0 = torch.zeros(new_batch_size, self.config['unimodal_context']['audio_lstm_hidden_size']).unsqueeze(
            0).to(
            self.device)
        pose_c0 = torch.zeros(new_batch_size, self.config['unimodal_context']['pose_lstm_hidden_size']).unsqueeze(0).to(
            self.device)

        text_out, (text_hn, text_cn) = self.t_lstm(x_t, (text_h0, text_c0))
        audio_out, (audio_hn, audio_cn) = self.a_lstm(x_a, (audio_h0, audio_c0))
        pose_out, (pose_hn, pose_cn) = self.p_lstm(x_p, (pose_h0, pose_c0))

        text_result = torch.reshape(text_hn, (batch_size, context_length, -1))
        audio_result = torch.reshape(audio_hn, (batch_size, context_length, -1))
        pose_result = torch.reshape(pose_hn, (batch_size, context_length, -1))

        return text_result, audio_result, pose_result


class MultimodalContextNetwork(nn.Module):
    def __init__(self, config):
        super(MultimodalContextNetwork, self).__init__()

        self.config = config

        self.text_dropout = nn.Dropout(config['multimodal_context']['text_dropout'])
        self.audio_dropout = nn.Dropout(config['multimodal_context']['audio_dropout'])
        self.pose_dropout = nn.Dropout(config['multimodal_context']['pose_dropout'])

        self.text_fc = nn.Linear(config['multimodal_context']['text_input'],
                                 config['multimodal_context']['text_mmcn_hidden_size'])
        self.audio_fc = nn.Linear(config['multimodal_context']['audio_input'],
                                  config['multimodal_context']['audio_mmcn_hidden_size'])
        self.pose_fc = nn.Linear(config['multimodal_context']['pose_input'],
                                 config['multimodal_context']['pose_mmcn_hidden_size'])

        self.self_attention = Transformer(d_input=config['multimodal_context']['d_input'],
                                          d_target=config['multimodal_context']['d_target'],
                                          max_length=config['multimodal_context']['max_length'],
                                          d_model=config['multimodal_context']['d_model'],
                                          n_head=config['multimodal_context']['n_head'],
                                          n_layers=config['multimodal_context']['n_layers'],
                                          d_feedforward=config['multimodal_context']['d_feedforward'],
                                          d_key=config['multimodal_context']['d_key'],
                                          d_value=config['multimodal_context']['d_value'],
                                          dropout=config['multimodal_context']['dropout']
                                          )

        self.dropout = nn.Dropout(config['multimodal_context']['dropout'])

    def forward(self, text_uni, audio_uni, pose_uni):
        reshaped_text_uni = text_uni.reshape((text_uni.shape[0], -1))
        reshaped_audio_uni = audio_uni.reshape((audio_uni.shape[0], -1))
        reshaped_pose_uni = pose_uni.reshape((pose_uni.shape[0], -1))

        mcn_text = self.text_dropout(self.text_fc(reshaped_text_uni))
        mcn_audio = self.audio_dropout(self.audio_fc(reshaped_audio_uni))
        mcn_pose = self.pose_dropout(self.pose_fc(reshaped_pose_uni))

        concat = torch.cat([text_uni, audio_uni, pose_uni], dim=2)

        mcn_mem = self.dropout(self.self_attention(concat)).squeeze(0)

        return mcn_mem, mcn_text, mcn_audio, mcn_pose


class HumorClassifier(nn.Module):
    def __init__(self, config):
        super(HumorClassifier, self).__init__()
        self.config = config

        self.unimodal_context = UnimodalContextNetwork(config)
        self.multimodal_context = MultimodalContextNetwork(config)

        self.ff = nn.Linear(config['multimodal_context']['d_target'], 1)

    def forward(self, x_t, x_a, x_p):
        uni_t, uni_a, uni_p = self.unimodal_context(x_t, x_a, x_p)
        mcn_mem, mcn_text, mcn_audio, mcn_pose = self.multimodal_context(uni_t, uni_a, uni_p)

        pred = self.ff(mcn_mem)
        return pred
