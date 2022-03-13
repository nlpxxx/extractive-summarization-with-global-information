import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

import distributed
# import onmt
import torch.nn as nn

from models.reporter import ReportMgr
from models.stats import Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str
import torch.nn.functional as F
from itertools import combinations


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model,
                  optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"


    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, pred, labels, weight=None):   # return shape: batch * sent
        """
        :param pred: batch * sent
        :param labels: batch * sent
        :param weight: batch
        :return: batch * sent
        """

        # loss = -(weight.unsqueeze(1) * labels * torch.log(pred) + (1 - labels) * torch.log(1 - pred))
        batch_size, sent_num = pred.size()
        loss = torch.zeros(batch_size, sent_num)
        for i in range(batch_size):
            _loss = torch.zeros(sent_num)
            for j in range(sent_num):
                if labels[i][j] == 0:
                    __loss = -(torch.log(1 - pred[i][j]))
                else:
                    __loss = -(weight[i] * torch.log(pred[i][j]))
                _loss[j] = __loss
            loss[i] = _loss

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, true):
        """
        :param pred: B * N
        :param true: B * N
        :return: loss: B * N
        """
        pred = pred + 1e-18
        # return -1.0 * true * ((1-pred)**self.gamma) * pred.log() \
        #        - 1.0 * (1 - true) * (pred**self.gamma) * (1-pred).log()
        return -self.alpha * true * ((1-pred)**self.gamma) * pred.log() \
               - (1 - self.alpha) * (1 - true) * (pred**self.gamma) * (1-pred).log()


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self,  args, model,  optim,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = torch.nn.BCELoss(reduction='none')
        # self.criterion = torch.nn.MSELoss(reduction="sum")
        # self.loss = FocalLoss()

        self.num_negative = args.num_negative
        self.margin = args.margin
        # self.tau = args.tau

        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()
            
    def _ntm_loss(self, doc_emb, sent_emb, mask_cls, labels, ntm_out, ortho_reg=0.1, temp=0.05, fac=1.0):
        def _reconstruction_loss(raw, rec, neg, margin, temp=0.1):
            # com_mat = torch.cat((raw.unsqueeze(1), neg), 1)
            # siliarity_mat = F.cosine_similarity(rec.unsqueeze(1), com_mat, dim=-1, eps=1e-08)
            # # siliarity_mat = siliarity_mat / temp
            # y = torch.ones_like(siliarity_mat) * 1e-18
            # siliarity_mat = torch.where(siliarity_mat == 0, y, siliarity_mat)
            # loss_fnt = nn.CrossEntropyLoss(reduction='none')
            # target = torch.zeros(siliarity_mat.size(0)).long().cuda()
            # rec_loss = loss_fnt(siliarity_mat, target)
            #
            # return rec_loss
            positive_dot_products = torch.matmul(raw.unsqueeze(1), rec.unsqueeze(2)).squeeze(-1) \
                                    # + torch.matmul(rec_emb.unsqueeze(1), pos.transpose(1,2)).sum(-1)
            # positive_dot_products = positive_dot_products / temp
            negative_dot_products = torch.matmul(neg, rec.unsqueeze(2)).squeeze()
            # negative_dot_products = negative_dot_products / temp
            negative_dot_products = torch.sum(negative_dot_products, dim=-1).unsqueeze(-1)

            if len(positive_dot_products.shape) < 2:
                positive_dot_products.unsqueeze(-1)
            if len(negative_dot_products) < 2:
                negative_dot_products.unsqueeze(-1)

            reconstruction_triplet_loss = torch.sum(margin - positive_dot_products + negative_dot_products, dim=1)
            max_margin = torch.max(reconstruction_triplet_loss, torch.zeros_like(reconstruction_triplet_loss))

            return max_margin  # batch

        def _reconstruction_DLC_loss(raw, rec, neg, temp=0.1, margin=1):
            cross_distance = raw @ rec.t()
            pos_loss = -torch.diag(cross_distance)   # B
            # pos_loss = -torch.diag(cross_distance) / temp   # B
            neg_prod = (rec.unsqueeze(1) @ neg.transpose(1, 2)).squeeze() # B * N
            # neg_prod = (rec.unsqueeze(1) @ neg.transpose(1, 2) / temp).squeeze() # B * N
            y = torch.ones_like(neg_prod) * 1e-18
            neg_sim = torch.where(neg_prod == 0, y, neg_prod)
            neg_loss = neg_sim.logsumexp(dim=-1)    # B
            dcl_loss = torch.max(pos_loss + neg_loss, torch.zeros_like(pos_loss))

            return dcl_loss

        def _ortho_regularizer(topic_embeddings):
            return torch.norm(
                torch.matmul(topic_embeddings, topic_embeddings.t()) - torch.eye(topic_embeddings.size(0)).cuda())  # 2范数

            # asp_norm = torch.norm(topic_embeddings, dim=-1, keepdim=True)  # aspect_dim * 1
            # asp_norm = topic_embeddings / asp_norm
            # asp_norm_loss = torch.matmul(asp_norm, asp_norm.t()) - torch.eye(topic_embeddings.size(0)).cuda()
            # asp_norm_loss = asp_norm_loss.abs().sum()
            # return asp_norm_loss

        def _kl_div(ntm_out):
            _, __, prior_mean, prior_variance, posterior_mean, posterior_log_variance = ntm_out     # B * N * K

            posterior_mean = posterior_mean.view(-1, posterior_mean.size(-1))
            posterior_log_variance = posterior_log_variance.view(-1, posterior_log_variance.size(-1))
            posterior_variance = torch.exp(posterior_log_variance)
            # var division term
            # var_division = torch.sum(posterior_variance / prior_variance, dim=-1)    # B * N
            # var_division = torch.sum(var_division, dim=1)   # B

            var_division = torch.sum(posterior_variance / prior_variance, dim=1)    # B

            # diff means term
            # diff_means = prior_mean - posterior_mean    # B * N * K
            # diff_term = torch.sum((diff_means * diff_means) / prior_variance, dim=-1)  # B * N
            # diff_term = torch.sum(diff_term, dim=1)    # B

            diff_means = prior_mean - posterior_mean    # B * K
            diff_term = torch.sum((diff_means * diff_means) / prior_variance, dim=1)  # B

            # logvar det division term
            logvar_det_division = \
                prior_variance.log().sum() - posterior_log_variance.sum(dim=1)  # B

            # logvar_det_division = \
            #     prior_variance.log().sum() - posterior_log_variance.sum(dim=1)  # B

            # combine terms
            KL = 0.5 * (
                    var_division + diff_term - prior_mean.size(-1) + logvar_det_division)   # (B * N)

            return KL

        def _reconstruction_cl_loss(raw, rec, mask, temp=0.05):
            hid = raw.size(2)
            sim = F.cosine_similarity(rec.view(-1, 1, hid), raw.view(1, -1, hid), dim=-1, eps=1e-08)
            sim = sim / temp
            y = torch.ones_like(sim) * 1e-18
            sim = torch.where(sim==0, y, sim)
            loss_fnt = nn.CrossEntropyLoss(reduction='none')
            label_pos = torch.arange(sim.size(0)).cuda()
            rec_loss = loss_fnt(sim, label_pos)    # (B * N)
            mask = mask.view(-1)
            rec_loss = (rec_loss * mask.float()).sum()
            
            return rec_loss

        rec_emb, topic_embeddings = ntm_out[0], ntm_out[1]
        batch_size, hid_size = doc_emb.size()

        num_negative = self.num_negative
        if self.num_negative > mask_cls.size(1):
            num_negative = mask_cls.size(1)
        real_num_sent = torch.sum(mask_cls, dim=1)
        neg_emb = torch.zeros(batch_size, num_negative, hid_size).cuda()
        # pos_emb = torch.zeros(batch_size, 3, hid_size).cuda()
        for i in range(batch_size):
            rev_labels = (labels[i] == 0)[:real_num_sent[i]]
            neg_idx = torch.nonzero(rev_labels)
            idx_perm = torch.randperm(len(neg_idx))[:num_negative]
            rand_sel = neg_idx[idx_perm]
            for j in range(num_negative):
                if j < len(rand_sel):
                    neg_emb[i][j] = sent_emb[i][rand_sel[j]]  # neg_idx[idx_perm]是非label句子的下标
                else:
                    neg_emb[i][j] = sent_emb[i][j]

            # # 正例
            # c = 0
            # for ind, j in enumerate(labels[i]):
            #     if j == 1:
            #         pos_emb[i][c] = sent_emb[i][ind]
            #         c += 1

        # max_margin = _reconstruction_DLC_loss(doc_emb, rec_emb, neg_emb)
        max_margin = _reconstruction_loss(doc_emb, rec_emb, neg_emb, margin=self.margin)
        regular = ortho_reg * _ortho_regularizer(topic_embeddings)
        # rec_loss = _reconstruction_cl_loss(sent_emb, rec_emb, mask_cls, temp)

        # kl = _kl_div(ntm_out).view(rec_emb.size(0), rec_emb.size(1)) * mask_cls
        # kl = kl.sum()
        # kl = _kl_div(ntm_out).sum()

        # return rec_loss + kl * fac
        # return torch.sum(rec_loss + regular)
        return torch.sum(max_margin + regular)


    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        # step =  self.optim._step + 1
        step =  self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step >= self.args.save_from and step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:

                src = batch.src
                tgt = batch.tgt
                labels = batch.labels
                # segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_tgt = batch.mask_tgt
                mask_cls = batch.mask_cls

                # sent_scores, mask = self.model(src, clss, labels, tgt, mask, mask_tgt, mask_cls)
                sent_scores, mask, doc, sent, ntm = self.model(src, clss, labels, tgt, mask, mask_tgt, mask_cls)

                # weighted = (torch.sum(mask, dim=1) - torch.sum(labels, dim=1)) / (
                #         torch.sum(labels, dim=1) + 1e-18)
                # wgt_labels = weighted.unsqueeze(-1) * labels
                # loss = self.loss(sent_scores, wgt_labels.float()).cuda()

                loss = self.loss(sent_scores, labels.float())   # shape of loss : batch * sent
                # ntm_loss = self._ntm_loss(doc, ntm[0], sent, mask_cls, labels, ntm[1], ortho_reg=0.1)

                # weighted = (torch.sum(mask, dim=1) - torch.sum(labels, dim=1)) / torch.sum(labels, dim=1)
                # loss = self.loss(sent_scores, labels.float()).cuda()  # reduction = none, so : batch * sent_num

                loss = (loss * mask.float()).sum()
                # cos_loss = 0
                # for i in range(len(batch)):
                #     loss_ = F.cosine_similarity(mean_doc[i], mean_sum[i], dim=0)
                #     cos_loss_ = (1 - loss_).sum()
                #     if torch.isnan(cos_loss_):
                #         continue
                #     else:
                #         cos_loss += cos_loss_
                # loss += cos_loss
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        if not os.path.exists(self.args.result_path):
            os.makedirs(self.args.result_path)
        can_path = os.path.join(self.args.result_path, 'step%d.candidate' % (step))
        gold_path = os.path.join(self.args.result_path, 'step%d.gold' % (step))
        compar_path = os.path.join(self.args.result_path, 'step%d.labels' % (step))

        correct_cnt = 0
        total_sent = 0
        label_sent = 0
        sel_sent = 0

        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with open(compar_path, 'w') as compar:

                    with torch.no_grad():
                        for batch in test_iter:
                            src = batch.src
                            tgt = batch.tgt
                            labels = batch.labels
                            # segs = batch.segs
                            clss = batch.clss
                            mask = batch.mask
                            mask_tgt = batch.mask_tgt
                            mask_cls = batch.mask_cls


                            gold = []
                            pred = []

                            if (cal_lead):
                                selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                            elif (cal_oracle):
                                selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                                range(batch.batch_size)]
                            else:
                                # sent_scores, mask, mean_doc, top_vec = self.model(src, clss, labels, tgt, mask, mask_tgt,
                                #                                                   mask_cls, need_art=True, is_test=True)
                                # sent_scores, mask = self.model(src, clss, labels, tgt, mask, mask_tgt, mask_cls)
                                sent_scores, mask, doc, sent, ntm = self.model(src, clss, labels, tgt, mask, mask_tgt,
                                                                               mask_cls)

                                # weighted = (torch.sum(mask, dim=1) - torch.sum(labels, dim=1)) / (
                                #             torch.sum(labels, dim=1) + 1e-18)
                                # wgt_labels = weighted.unsqueeze(-1) * labels
                                # loss = self.loss(sent_scores, wgt_labels.float()).cuda()

                                loss = self.loss(sent_scores, labels.float())
                                loss = (loss * mask.float()).sum()

                                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                                stats.update(batch_stats)
                                sent_scores = sent_scores + mask.float()
                                sent_scores = sent_scores.cpu().data.numpy()    # batch_size * sent_num
                                selected_ids = np.argsort(-sent_scores, 1)  # batch_size * sent_num
                            # selected_ids = np.sort(selected_ids,1)

                            for i, idx in enumerate(selected_ids):
                                ids_rec_ = []
                                _pred = []
                                if(len(batch.src_str[i])==0):
                                    continue
                                for j in selected_ids[i][:len(batch.src_str[i])]:
                                    if(j>=len( batch.src_str[i])):
                                        continue
                                    candidate = batch.src_str[i][j].strip()
                                    if(self.args.block_trigram):
                                        if(not _block_tri(candidate,_pred)):
                                            _pred.append(candidate)
                                            ids_rec_.append(j)
                                    else:
                                        _pred.append(candidate)
                                        ids_rec_.append(j)

                                    if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                        break

                                _pred = '<q>'.join(_pred)
                                if(self.args.recall_eval):
                                    _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                                # pred.append(final_pred)
                                pred.append(_pred)
                                gold.append(batch.tgt_str[i])

                                total_sent += sum(mask[i])
                                label_sent += sum(labels[i])
                                sel_sent += len(ids_rec_)
                                for rk, ii in enumerate(labels[i]):
                                    if ii == 1 and (rk in ids_rec_):
                                        correct_cnt += 1
                                temp_sel = torch.zeros_like(labels[i])
                                temp_sel[ids_rec_] = 1
                                compar.write('labels:   ' + str(labels[i].cpu().numpy().tolist()) + '\n')
                                compar.write('selected: ' + str(temp_sel.cpu().numpy().tolist()) + '\n')
                            for i in range(len(gold)):
                                save_gold.write(gold[i].strip()+'\n')
                            for i in range(len(pred)):
                                save_pred.write(pred[i].strip()+'\n')
                                
        logger.info(f'total sentences: {total_sent}, labeled sentences: {label_sent}, selected sentence: {sel_sent}')
        logger.info('selected recall: {:.2%}'.format(correct_cnt/label_sent))
        logger.info('selected precision: {:.2%}'.format(correct_cnt/sel_sent))
        logger.info('label proportion: {:.2%}'.format(label_sent/total_sent))

        if(step!=-1 and self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)

        return stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            tgt = batch.tgt
            labels = batch.labels   # batch * max_labels_len
            # segs = batch.segs
            clss = batch.clss
            mask = batch.mask
            mask_tgt = batch.mask_tgt
            mask_cls = batch.mask_cls

            # mask： mask_cls, sent_scores: batch * sent_num
            # sent_scores, mask, mean_doc, mean_sum = self.model(src, clss, labels, tgt,  mask, mask_tgt, mask_cls)
            sent_scores, mask, doc, sent, ntm = self.model(src, clss, labels, tgt,  mask, mask_tgt, mask_cls)
            # ntm_loss = self._ntm_loss(sent, mask_cls, labels, ntm, temp=self.tau, fac=1)
            # ntm_loss = self._ntm_loss(doc, sent, mask_cls, labels, ntm, ortho_reg=0.1, temp=self.tau)
            ntm_loss = self._ntm_loss(doc, sent, mask_cls, labels, ntm, ortho_reg=0.1)

            # ntm_loss = self.criterion(ntm, torch.zeros_like(ntm))

            loss = self.loss(sent_scores, labels.float()).cuda()   # reduction = none, so : batch * sent_num

            # weighted = (torch.sum(mask, dim=1) - torch.sum(labels, dim=1)) / (torch.sum(labels, dim=1)+1e-18)
            # wgt_labels = weighted.unsqueeze(-1) * labels
            # ones = torch.ones_like(wgt_labels)
            # wgt_labels = torch.where(wgt_labels == 0, ones, wgt_labels)
            # loss *= wgt_labels
            loss = (loss*mask.float()).sum()

            loss += ntm_loss

            # if torch.isnan(loss):
            #     print('the cls of the nan: ', clss)
            #     print('the label of the nan: ', labels)
            # numel() Returns the total number of elements in the input tensor.
            (loss/loss.numel()).backward()
            # print('sent_scores : ', sent_scores.requires_grad)
            # print('sent_scores : ', sent_scores.grad)
            # print('mean_doc : ', mean_doc.grad)
            # print('mean_sum : ', mean_sum.grad)

            # loss.div(float(normalization)).backward()

            batch_stats = Statistics(float(loss.cpu().data.numpy()),
                                     batch.batch_size, float(ntm_loss.cpu().data.numpy()))


            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
