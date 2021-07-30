from torch import nn
from torchvision import models
import torch


class BaseGoalPredictor(nn.Module):
    def __init__(self):
        super(BaseGoalPredictor, self).__init__()
        self.hist_enc = nn.Sequential(
            nn.Linear(2*3*8, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.Dropout(0.3),
        )

        self.att = torch.nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.pose_embed = nn.Linear(6 * 8, 32)
        self.q_emb = nn.Linear(32, 64)
        self.k_emb = nn.Linear(32, 64)
        self.v_emb = nn.Linear(32, 64)

        self.decoder = nn.Sequential(
            nn.Linear(64+64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2),
            nn.Dropout(0.0),
        )

    def encode_edge_att(self, self_hist, neighbours):
        bs = self_hist.shape[0]
        num_n = neighbours.shape[1]
        if num_n == 0:
            neighbours = torch.ones(bs, 1, 6)
        neighbours_enc = self.pose_embed(neighbours.reshape(bs, num_n, -1))
        pe = self.pose_embed(self_hist.reshape(bs, -1))  # bs, num_n, 32
        q = self.q_emb(pe).unsqueeze(1)
        k = self.k_emb(neighbours_enc)
        v = self.v_emb(neighbours_enc)
        attn_output, attn_output_weights = self.att(q, k, v)
        return attn_output[:,0,:]

    def forward(self, self_hist, neigb):
        pred_enc = self.hist_enc(self_hist.reshape(-1, 6*8))
        neigb_enc = self.encode_edge_att(self_hist, neigb)
        predictions = self.decoder(torch.cat((pred_enc, neigb_enc), dim=1))
        return predictions




class AttGoalPredictor(nn.Module):
    def __init__(self):
        super(AttGoalPredictor, self).__init__()
        self.hist_enc = nn.Sequential(
            nn.Linear(2*3*8, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.Dropout(0.3),
        )

        self.att = torch.nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.pose_embed = nn.Linear(6 * 8, 32)
        self.q_emb = nn.Linear(32, 64)
        self.k_emb = nn.Linear(32, 64)
        self.v_emb = nn.Linear(32, 64)

        self.att_dec = torch.nn.MultiheadAttention(embed_dim=32, num_heads=8)
        self.q_dec = nn.Linear(128, 32)
        self.k_dec = nn.Linear(128, 32)
        self.v_dec = nn.Linear(128, 32)

        self.dec = nn.Linear(32, 2)

    def encode_edge_att(self, self_hist, neighbours):
        bs = self_hist.shape[0]
        num_n = neighbours.shape[1]
        if num_n == 0:
            neighbours = torch.ones(bs, 1, 6)
        neighbours_enc = self.pose_embed(neighbours.reshape(bs, num_n, -1))
        pe = self.pose_embed(self_hist.reshape(bs, -1))  # bs, num_n, 32
        q = self.q_emb(pe).unsqueeze(1)
        k = self.k_emb(neighbours_enc)
        v = self.v_emb(neighbours_enc)
        attn_output, attn_output_weights = self.att(q, k, v)
        return attn_output[:, 0, :]

    def forward(self, self_hist, neigb):
        pred_enc = self.hist_enc(self_hist.reshape(-1, 6*8))
        neigb_enc = self.encode_edge_att(self_hist, neigb)
        encoded_vec = torch.cat((pred_enc, neigb_enc), dim=1)
        q = self.q_dec(encoded_vec).unsqueeze(1)
        k = self.k_dec(encoded_vec).unsqueeze(1)
        v = self.v_dec(encoded_vec).unsqueeze(1)
        attn_output, _ = self.att_dec(q, k, v)
        predictions = self.dec(attn_output)
        return predictions



class FullAttGoalPredictor(nn.Module):
    def __init__(self):
        super(FullAttGoalPredictor, self).__init__()
        self.q_hist_emb = nn.Sequential(nn.Linear(6, 32))
        self.k_hist_emb = nn.Sequential(nn.Linear(6, 32))
        self.v_hist_emb = nn.Sequential(nn.Linear(6, 32))
        self.hist_enc = torch.nn.MultiheadAttention(embed_dim=32, num_heads=4, dropout=0.6)
        self.att_emb = nn.Sequential(nn.Linear(32, 64), nn.BatchNorm1d(64))

        self.att = torch.nn.MultiheadAttention(embed_dim=64, num_heads=8, dropout=0.6)
        self.pose_embed = nn.Linear(6 * 8, 32)
        self.q_emb = nn.Sequential(nn.Linear(32, 64))
        self.k_emb = nn.Sequential(nn.Linear(32, 64))
        self.v_emb = nn.Sequential(nn.Linear(32, 64))

        self.att_dec = torch.nn.MultiheadAttention(embed_dim=64, num_heads=8, dropout=0.6)
        self.q_dec = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64))
        self.k_dec = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64))
        self.v_dec = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64))

        self.dec = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.Linear(8, 2),
        )


    def encode_edge_att(self, self_hist, neighbours):
        bs = self_hist.shape[0]
        num_n = neighbours.shape[1]
        if num_n == 0:
            neighbours = torch.ones(bs, 1, 6)
        neighbours_enc = self.pose_embed(neighbours.reshape(bs, num_n, -1))
        pe = self.pose_embed(self_hist.reshape(bs, -1))  # bs, num_n, 32
        q = self.q_emb(pe).unsqueeze(1)
        k = self.k_emb(neighbours_enc)
        v = self.v_emb(neighbours_enc)
        attn_output, attn_output_weights = self.att(q, k, v)
        return attn_output[:, 0, :]

    def forward(self, self_hist, neigb):
        q = self.q_hist_emb(self_hist)
        k = self.k_hist_emb(self_hist)
        v = self.v_hist_emb(self_hist)
        pred_enc, _ = self.hist_enc(q, k, v)
        pred_enc = self.att_emb(pred_enc[:, 0, :])
        neigb_enc = self.encode_edge_att(self_hist, neigb)
        encoded_vec = torch.cat((pred_enc, neigb_enc), dim=1)
        q = self.q_dec(encoded_vec).unsqueeze(1)
        k = self.k_dec(encoded_vec).unsqueeze(1)
        v = self.v_dec(encoded_vec).unsqueeze(1)
        attn_output, _ = self.att_dec(q, k, v)
        predictions = self.dec(attn_output[:,0,:])
        return predictions

class LSTM_simple(nn.Module):
    def __init__(self):
        super(LSTM_simple, self).__init__()
        self.hist_emb = nn.Sequential(nn.Linear(6, 32))
        self.hist_enc = nn.LSTM(32, 64, batch_first=True, dropout=0.3)
        self.dec = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.Linear(8, 2),
        )


    def forward(self, self_hist, neigb):
        hist_emb = self.hist_emb(self_hist)
        encoded_hist, _ = self.hist_enc(hist_emb)
        predictions = self.dec(encoded_hist[:, -1, :])
        # v = self.v_hist_emb(self_hist)
        # pred_enc, _ = self.hist_enc(q, k, v)
        # pred_enc = self.att_emb(pred_enc[:, 0, :])
        # neigb_enc = self.encode_edge_att(self_hist, neigb)
        # encoded_vec = torch.cat((pred_enc, neigb_enc), dim=1)
        # q = self.q_dec(encoded_vec).unsqueeze(1)
        # k = self.k_dec(encoded_vec).unsqueeze(1)
        # v = self.v_dec(encoded_vec).unsqueeze(1)
        # attn_output, _ = self.att_dec(q, k, v)
        # predictions = self.dec(attn_output[:,0,:])
        # predictions = torch.cumsum(predictions, dim=1)
        return predictions

class LSTM_att(nn.Module):
    def __init__(self):
        super(LSTM_att, self).__init__()
        self.hist_emb = nn.Sequential(nn.Linear(6, 32))
        self.hist_enc = nn.LSTM(32, 64, batch_first=True, dropout=0.3)

        self.att = torch.nn.MultiheadAttention(embed_dim=64, num_heads=8, dropout=0.6)
        self.pose_embed = nn.Linear(6 * 8, 32)
        self.q_emb = nn.Sequential(nn.Linear(64, 64))
        self.k_emb = nn.Sequential(nn.Linear(32, 64))
        self.v_emb = nn.Sequential(nn.Linear(32, 64))
        self.dec = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.Linear(8, 2),
        )

    def forward(self, self_hist, neigb):
        hist_emb = self.hist_emb(self_hist)
        encoded_hist, _ = self.hist_enc(hist_emb)
        enc_scene = self.encode_edge_att(encoded_hist, neigb)
        predictions = self.dec(enc_scene)
        # v = self.v_hist_emb(self_hist)
        # pred_enc, _ = self.hist_enc(q, k, v)
        # pred_enc = self.att_emb(pred_enc[:, 0, :])
        # neigb_enc = self.encode_edge_att(self_hist, neigb)
        # encoded_vec = torch.cat((pred_enc, neigb_enc), dim=1)
        # q = self.q_dec(encoded_vec).unsqueeze(1)
        # k = self.k_dec(encoded_vec).unsqueeze(1)
        # v = self.v_dec(encoded_vec).unsqueeze(1)
        # attn_output, _ = self.att_dec(q, k, v)
        # predictions = self.dec(attn_output[:,0,:])
        # predictions = torch.cumsum(predictions, dim=1)
        return predictions

    def encode_edge_att(self, self_hist, neighbours):
        bs = self_hist.shape[0]
        num_n = neighbours.shape[1]
        if num_n == 0:
            neighbours = torch.ones(bs, 1, 6)
        neighbours_enc = self.pose_embed(neighbours.reshape(bs, num_n, -1))
        # pe = self.pose_embed(self_hist.reshape(bs, -1))  # bs, num_n, 32
        q = self.q_emb(self_hist)
        k = self.k_emb(neighbours_enc)
        v = self.v_emb(neighbours_enc)
        attn_output, attn_output_weights = self.att(q, k, v)
        return attn_output[:, 0, :]

if __name__ == '__main__':
    model = LSTM_simple()
    bs = 2
    n_neighb = 10
    self_hist = torch.rand(bs, 8, 6)
    neigb = torch.rand(bs, 10, 8, 6)
    predictions = model(self_hist, neigb)
    print (predictions.shape)