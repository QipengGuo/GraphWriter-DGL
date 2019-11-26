import torch
from modules import MSA, BiLSTM, GraphTrans  
from utlis import *
from torch import nn
import time
            
class GraphWriter(nn.Module):
    def __init__(self, args):
        super(GraphWriter, self).__init__()
        self.args = args
        if args.share_vocab:
            tmp_emb = nn.Embedding(len(args.text_vocab), args.nhid)
        if args.title:
            self.title_emb = tmp_emb if args.share_vocab else nn.Embedding(len(args.text_vocab), args.nhid)
            self.title_enc = BiLSTM(args, enc_type='title')
            self.title_attn = MSA(args)
        self.ent_emb = tmp_emb if args.share_vocab else nn.Embedding(len(args.text_vocab), args.nhid)
        self.tar_emb = tmp_emb if args.share_vocab else nn.Embedding(len(args.text_vocab), args.nhid)
        self.rel_emb = nn.Embedding(len(args.rel_vocab), args.nhid)
        self.decode_lstm = nn.LSTMCell(args.dec_ninp, args.nhid)
        self.ent_enc = BiLSTM(args, enc_type='entity')
        self.graph_enc = GraphTrans(args)
        self.ent_attn = MSA(args)
        self.copy_attn = MSA(args, mode='copy')
        self.copy_fc = nn.Linear(args.dec_ninp, 1)
        self.pred_v_fc = nn.Linear(args.dec_ninp, len(args.text_vocab))

    def enc_forward(self, batch, ent_mask, ent_text_mask, ent_len, rel_mask, title_mask):
        title_enc = None
        if self.args.title:
            title_enc = self.title_enc(self.title_emb(batch['title']), title_mask)
        ent_enc = self.ent_enc(self.ent_emb(batch['ent_text']), ent_text_mask, ent_len = batch['ent_len'])
        rel_emb = self.rel_emb(batch['rel'])
        g_ent, g_root = self.graph_enc(ent_enc, ent_mask, ent_len, rel_emb, rel_mask, batch['graph'])
        return g_ent, g_root, title_enc, ent_enc

    def forward(self, batch, beam_size=None):
        ent_mask = len2mask(batch['ent_len'], self.args.device)
        ent_text_mask = batch['ent_text']==0
        rel_mask = batch['rel']==0 # 0 means the <PAD>
        title_mask = batch['title']==0
        g_ent, g_root, title_enc, ent_enc = self.enc_forward(batch, ent_mask, ent_text_mask, batch['ent_len'], rel_mask, title_mask)

        _h, _c = g_root, g_root#.clone().detach()
        ctx = torch.zeros_like(g_root)
        if self.args.title:
            attn = self.title_attn(_h, title_enc, mask=title_mask)
            ctx = torch.cat([ctx, attn], 1)
        if beam_size is None:
            outs = []
            tar_inp = self.tar_emb(batch['text'].transpose(0,1))
            for t, xt in enumerate(tar_inp):
                _xt = torch.cat([ctx, xt], 1)
                _h, _c = self.decode_lstm(_xt, (_h, _c))
                ctx = self.ent_attn(_h, g_ent, mask=ent_mask)
                if self.args.title:
                    attn = self.title_attn(_h, title_enc, mask=title_mask)
                    ctx = torch.cat([ctx, attn], 1)
                outs.append(torch.cat([_h, ctx], 1)) 
            outs = torch.stack(outs, 1)
            copy_gate = torch.sigmoid(self.copy_fc(outs))
            EPSI = 1e-6
            pred_v = torch.log(copy_gate+EPSI) + torch.log_softmax(self.pred_v_fc(outs), -1)
            pred_c = torch.log((1. - copy_gate)+EPSI) + torch.log_softmax(self.copy_attn(outs, ent_enc, mask=ent_mask), -1)
            pred = torch.cat([pred_v, pred_c], -1)
            return pred
        else:
            # force greedy
            device = g_ent.device
            B = g_ent.shape[0]
            ent_type = batch['ent_type'].view(B, -1)
            beam_seq = (torch.ones(B,).long().to(device) * self.args.text_vocab('<BOS>')).unsqueeze(1)
            for t in range(self.args.beam_max_len):
                _inp = replace_ent(beam_seq[:,-1], ent_type, len(self.args.text_vocab))
                xt = self.tar_emb(_inp)
                _xt = torch.cat([ctx, xt], 1)
                _h, _c = self.decode_lstm(_xt, (_h, _c))
                ctx = self.ent_attn(_h, g_ent, mask=ent_mask)
                if self.args.title:
                    attn = self.title_attn(_h, title_enc, mask=title_mask)
                    ctx = torch.cat([ctx, attn], 1)
                _y = torch.cat([_h, ctx], 1)
                copy_gate = torch.sigmoid(self.copy_fc(_y))
                pred_v = torch.log(copy_gate) + torch.log_softmax(self.pred_v_fc(_y), -1)
                pred_c = torch.log((1. - copy_gate)) + torch.log_softmax(self.copy_attn(_y.unsqueeze(1), ent_enc, mask=ent_mask).squeeze(1), -1)
                pred = torch.cat([pred_v, pred_c], -1).view(B,-1)
                for ban_item in ['<BOS>', '<PAD>', '<UNK>']:
                    pred[:, self.args.text_vocab(ban_item)] = -1e8
                _, word = pred.max(-1)
                beam_seq = torch.cat([beam_seq, word.unsqueeze(1)], 1)
            return beam_seq, None
  
