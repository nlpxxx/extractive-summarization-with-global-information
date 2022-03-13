
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from torch.nn.init import xavier_uniform_

from models.encoder import TransformerInterEncoder, Classifier, RNNEncoder
from models.optimizers import Optimizer
from collections import OrderedDict


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class Bert(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()
        if(load_pretrained_bert):
            self.model = RobertaModel.from_pretrained('/roberta-base')
        else:
            self.model = RobertaModel(bert_config)

    def forward(self, x, mask):
        output = self.model(x, attention_mask=mask)
        encoded_layers = output.last_hidden_state
        return encoded_layers


class NeuralTopicModel(nn.Module):
    def __init__(self, num_topic, hid_dim,  activation='softplus', dropout=0.1, init_topic_matrix=None):
        super(NeuralTopicModel, self).__init__()
        self.ntopic = num_topic
        self.hid_dim = hid_dim
        # self.hidden_sizes = (self.hid_dim, 512, self.ntopic)
        self.hid_layer = nn.Linear(self.hid_dim, self.ntopic)
        self.softmax = nn.Softmax(dim=-1)
        # self.dropout = dropout
        self.topic_embeddings = nn.Parameter(torch.empty(size=(num_topic, hid_dim)))

        # topic_prior_mean = 0.0
        # self.prior_mean = torch.tensor([topic_prior_mean] * self.ntopic).cuda()
        # self.prior_mean = nn.Parameter(self.prior_mean)
        #
        # topic_prior_variance = 1. - (1. / self.ntopic)
        # self.prior_variance = torch.tensor([topic_prior_variance] * self.ntopic).cuda()
        # self.prior_variance = nn.Parameter(self.prior_variance)
        #
        # if activation == 'softplus':
        #     self.activation = nn.Softplus()
        # elif activation == 'relu':
        #     self.activation = nn.ReLU()
        #
        # self.hiddens = nn.Sequential(OrderedDict([
        #     ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), nn.Dropout(self.dropout), self.activation))
        #     for i, (h_in, h_out) in enumerate(zip(self.hidden_sizes[:-1], self.hidden_sizes[1:]))]))
        #
        # self.f_mu = nn.Linear(self.hidden_sizes[-1], self.ntopic)
        # self.f_sigma = nn.Linear(self.hidden_sizes[-1], self.ntopic)
        # self.dropout_enc = nn.Dropout(p=self.dropout)

        if init_topic_matrix is None:
            nn.init.xavier_uniform_(self.topic_embeddings)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, doc_emb):
        # x = self.hiddens(doc_emb)

        # x = self.dropout_enc(x)
        # mu = self.f_mu(x)
        # log_sigma = self.f_sigma(x)
        # topic_dist = self.softmax(self.reparameterize(mu, log_sigma))

        # topic_dist = self.softmax(x)
        topic_dist = self.softmax(self.hid_layer(doc_emb))

        rec_emb = torch.matmul(topic_dist, self.topic_embeddings)   # batch_size *  hid_size

        # return rec_emb, self.topic_embeddings,  self.prior_mean, self.prior_variance, mu, log_sigma
        return rec_emb, self.topic_embeddings


class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, bert_config = None):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.temp_dir, load_pretrained_bert, bert_config)
        self.ntm = NeuralTopicModel(args.ntopic, self.bert.model.config.hidden_size)
        # self.linear_wgt = nn.Linear(self.bert.model.config.hidden_size, 1)
        if (args.encoder == 'classifier'):
            self.encoder = Classifier(self.bert.model.config.hidden_size)
        elif(args.encoder=='transformer'):
            self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size, args.ff_size, args.heads,
                                                   args.dropout, args.inter_layers)
        elif(args.encoder=='rnn'):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
                                      input_size=self.bert.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
        elif (args.encoder == 'baseline'):
            bert_config = RobertaConfig(self.bert.model.config.vocab_size, hidden_size=args.hidden_size,
                                     num_hidden_layers=6, num_attention_heads=8, intermediate_size=args.ff_size)
            self.bert.model = RobertaModel(bert_config)
            self.encoder = Classifier(self.bert.model.config.hidden_size)

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        self.to(device)
    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, clss, labels, tgt, mask, mask_tgt, mask_cls, need_art=False, is_test=False):

        top_vec = self.bert(x, mask)
        mean_doc = self.get_mean_doc(top_vec, x.size(0), clss, self.bert.model.config.hidden_size).to(self.device)
        mean_sent = self.get_mean_sent(top_vec, x.size(0), x, clss, self.bert.model.config.hidden_size).to(self.device)

        # mean_sent, mean_doc = self.wgt_vec(top_vec, clss, mask)

        ntm_output = self.ntm(mean_doc)
        # ntm_output = self.ntm(mean_sent)
        topic_vecs = ntm_output[0]  # B * N * H
        # topic_vecs = topic_vecs.unsqueeze(1).repeat(1, mean_sent.size(1), 1)

        # sent_scores = self.encoder(topic_vecs, mask_cls, topic_vecs).squeeze(-1)  # batch_size * sent_num
        sent_scores = self.encoder(mean_sent, mask_cls, topic_vecs).squeeze(-1)  # batch_size * sent_num

        return sent_scores, mask_cls, mean_doc, mean_sent, ntm_output

    def wgt_vec(self, x, clss, mask=None):
        batch_size, hid_size = x.size(0), x.size(-1)
        wgt = self.linear_wgt(x).squeeze(-1)   # B * S
        if mask is not None:
            wgt = wgt.masked_fill(~mask, -1e18)

        fnt = nn.Softmax(dim=-1)
        attn = fnt(wgt).unsqueeze(-1)
        doc_vec = torch.sum(attn * x, -2).to(self.device)
        sent_vec = torch.zeros(batch_size, clss.size(1), hid_size).to(self.device)
        for i in range(batch_size):
            sents = torch.zeros(clss.size(1), hid_size)
            for idx, j in enumerate(clss[i]):
                if idx == 0:    start = 0
                else:   start = clss[i][idx-1] + 1

                if (clss[i][idx] - clss[i][idx - 1]) == 1:  # </s></s>的情况
                    sents[idx] = attn[i, clss[i][idx], :] * x[i, clss[i][idx], :]
                else:
                    sents[idx] = torch.sum(attn[i, start:clss[i][idx]+1, :] * x[i, start:clss[i][idx]+1, :], -2)

                if clss[i][idx] == clss[i][-1] or clss[i][idx + 1] == 0:
                    break
            sent_vec[i] = sents

        return sent_vec, doc_vec

    def get_mean_sent(self, x, batch_size, src, clss, hid_size):
        mean_all = torch.zeros(batch_size, clss.size(1), hid_size)
        for i in range(batch_size):
            mean = torch.zeros(clss.size(1), hid_size)
            for idx, j in enumerate(clss[i]):
                if idx == 0:
                    mean[idx] = torch.mean(x[i, 0:(j+1), :], -2)
                else:
                    if (clss[i][idx] - clss[i][idx - 1]) == 1:  # </s></s>的情况
                        mean[idx] = x[i, clss[i][idx], :]
                    else:
                        mean[idx] = torch.mean(x[i, (clss[i][idx-1]+1):(j+1), :], -2)


                # 是否最后一个</s>
                if clss[i][idx] == clss[i][-1]:
                    # mean[idx + 1] = torch.mean(x[i, clss[i][-1]:(torch.nonzero(src[i])[-1]+1), :],
                    #                            -2)  # 应该是 clss[i][-1]：非mask的地方
                    break
                if clss[i][idx + 1] == 0:
                    # mean[idx + 1] = torch.mean(x[i, clss[i][idx + 1]:(torch.nonzero(src[i])[-1]+1), :],
                    #                            -2)  # 应该是 clss[i][-1]：非mask的地方
                    break
            mean_all[i] = mean

        return mean_all

    def get_mean_doc(self, x, batch_size, clss, hid_size):
        mean_doc = torch.zeros(batch_size, hid_size)

        for i in range(batch_size):
            mean_doc[i] = torch.mean(x[i, :(torch.nonzero(clss[i])[-1] + 1), :], -2)

        return mean_doc

    def get_mean_sum(self, tgt, batch_size, mask, hid_size):
        mean_sum = torch.zeros(batch_size, hid_size)
        for i in range(batch_size):
            nzero_pos = torch.nonzero(mask[i]).size(0)
            mean_sum[i] = torch.mean(tgt[i, :nzero_pos, :], 0)

        return mean_sum
        # mean_sum = torch.zeros(batch_size, hid_size)
        #
        # for i in range(batch_size):
        #     j = 0
        #     sent_num = torch.nonzero(clss[i]).size(0)
        #     mean_sum_ = torch.zeros(3, hid_size)
        #     for k, label in enumerate(labels[i]):
        #         if k == sent_num:
        #             break
        #         else:
        #             if label == 1:
        #                 if k == 0:
        #                     mean_sum_[j] = torch.mean(x[i, :(clss[i][k]+1), :], -2)
        #
        #                 else:
        #                     if (clss[i][k]-clss[i][k-1]) == 1:
        #                         mean_sum_[j] = x[i, clss[i][k], :]
        #                     else:
        #                         mean_sum_[j] = torch.mean(x[i, (clss[i][k-1]+1):(clss[i][k]+1), :], -2)
        #                 j += 1
        #     mean_sum_ = mean_sum_[:j]
        #     mean_sum[i] = torch.mean(mean_sum_, 0)
        #
        # return mean_sum