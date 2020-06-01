import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from . import tst_free
from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import json

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux

    def train(self):
        if self.args.tst_free:
            return self.train_tst()
        else:
            return self.train_original()

    def train_tst(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        # Clear evaluation file
        with open(osp.join(self.args.save_path, 'eval.jl'), 'w') as fp:
            pass

        # start FSL training
        label, label_aux = self.prepare_label()
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()

            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step += 1

                if torch.cuda.is_available():
                    data, gt_label = [_.cuda() for _ in batch]
                else:
                    data, gt_label = batch[0], batch[1]

                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                logits, reg_logits = self.para_model(data)
                if reg_logits is not None:
                    loss = F.cross_entropy(logits, label)
                    total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
                else:
                    loss = F.cross_entropy(logits, label)
                    total_loss = F.cross_entropy(logits, label)

                tl2.add(loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)

                # refresh start_tm
                start_tm = time.time()

                if args.debug_fast:
                    print('Debug fast, breaking training after 1 mini-batch')
                    break

            self.lr_scheduler.step()
            self.try_evaluate_tst(epoch)

            print('ETA:{}/{}'.format(
                self.timer.measure(),
                self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        #self.save_model('epoch-last')

    def try_evaluate_tst(self, epoch):
        args = self.args
        if self.train_epoch % args.eval_interval == 0:
            print('*'*32)

            stats = OrderedDict()
            stats['epoch'] = self.train_epoch

            for split, loader in [('valid', self.val_loader), ('test', self.test_loader)]:

                print('\nEpoch {} : Evaluating on {}'.format(self.train_epoch, split))

                vl, va, vap, metrics = self.evaluate(self.val_loader)
                split_stats = OrderedDict()
                split_stats['{}_SupervisedAcc'.format(split)] = va
                split_stats['{}_SupervisedAcc_interval'.format(split)] = vap
                split_stats['{}_SupervisedLoss'.format(split)] = vl
                for key, (val, ci) in metrics.items():
                    split_stats['{}_{}'.format(split, key)] = val
                    split_stats['{}_{}_interval'.format(split, key)] = ci

                stats.update(split_stats)

                text = ['{}={:.3f}'.format(key, val) for key, val in split_stats.items() if not key.endswith('_interval')]
                print(' | '.join(text))

                if split == 'valid':
                    # Do the best model thing
                    pass

            # dump the metrics
            with open(osp.join(self.args.save_path, 'eval.jl'), 'a') as fp:
                fp.write('{}\n'.format(json.dumps(stats)))

    def train_original(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        
        # start FSL training
        label, label_aux = self.prepare_label()
        #all_labels = torch.arange(args.way, dtype=torch.int16, device=label.device).repeat(args.shot + args.query)
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step += 1

                if torch.cuda.is_available():
                    data, gt_label = [_.cuda() for _ in batch]
                else:
                    data, gt_label = batch[0], batch[1]
               
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                logits, reg_logits = self.para_model(data)
                if reg_logits is not None:
                    loss = F.cross_entropy(logits, label)
                    total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
                else:
                    loss = F.cross_entropy(logits, label)
                    total_loss = F.cross_entropy(logits, label)
                    
                tl2.add(loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)    

                # refresh start_tm
                start_tm = time.time()

                if args.debug_fast:
                    print('Debug fast, breaking training after 1 mini-batch')
                    break

            self.lr_scheduler.step()
            self.try_evaluate(epoch)

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        metrics = OrderedDict()
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        all_labels = torch.arange(args.eval_way, device=label.device).repeat(args.eval_shot + args.eval_query)
        #print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
        #        self.trlog['max_acc_epoch'],
        #        self.trlog['max_acc'],
        #        self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                embeddings, logits = self.model(data, return_feature=True)

                # data contains both support and query sets (typically 25+75 for 5-shot 5-way 15-query)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc

                if args.tst_free:

                    embeddings_dict = self.model.get_embeddings_dict(embeddings, all_labels)

                    # TST-free part
                    for sinkhorn_reg_str in args.sinkhorn_reg:  # loop over all possible regularizations
                        sinkhorn_reg_float = float(sinkhorn_reg_str)
                        clustering_losses = tst_free.clustering_loss(embeddings_dict, sinkhorn_reg_float, 'wasserstein',
                                                                     temperature=np.sqrt(args.temperature),
                                                                     normalize_by_dim=False,
                                                                     clustering_iterations=20, sinkhorn_iterations=20,
                                                                     sinkhorn_iterations_warmstart=4,
                                                                     sanity_check=False)

                        for key, val in clustering_losses.items():
                            key += '_reg{}'.format(sinkhorn_reg_str)
                            metrics.setdefault(key, [])
                            metrics[key].append(val)

                if args.debug_fast:
                    print('Debug fast, breaking eval after 1 mini-batch')
                    record = record[:1]  # truncate summaries
                    break

        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        metric_summaries = {key: compute_confidence_interval(val) for key, val in metrics.items()}

        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        if args.tst_free:
            return vl, va, vap, metric_summaries
        else:
            return vl, va, vap

    def evaluate_test(self, use_max_tst=False):
        # restore model args
        args = self.args
        # evaluation mode

        if use_max_tst:
            assert args.tst_criterion != '', 'Please specify a criterion'
            fname = osp.join(self.args.save_path, 'max_tst_criterion.pth')
            criterion = args.tst_criterion
            max_acc_epoch = 'max_tst_criterion_epoch'
            max_acc = 'max_tst_criterion'
            max_acc_interval = 'max_tst_criterion_interval'
            test_acc = 'test_acc_at_max_criterion'
            test_acc_interval = 'test_acc_interval_at_max_criterion'
            test_loss = 'test_loss_at_max_criterion'
        else:
            fname = osp.join(self.args.save_path, 'max_acc.pth')
            criterion = 'SupervisedAcc'
            max_acc_epoch = 'max_acc_epoch'
            max_acc = 'max_acc'
            max_acc_interval = 'max_acc_interval'
            test_acc = 'test_acc'
            test_acc_interval = 'test_acc_interval'
            test_loss = 'test_loss'
        print('\nCriterion selected: {}'.format(criterion))
        print('Reloading model from {}'.format(fname))
        self.model.load_state_dict(torch.load(fname)['params'])

        self.model.eval()
        record = np.zeros((10000, 2)) # loss and acc
        metrics = defaultdict(list)  # all other metrics
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        all_labels = torch.arange(args.eval_way, device=label.device).repeat(args.eval_shot + args.eval_query)

        max_validation_str = 'Maximum value of valid_{} {:.4f} + {:.4f} reached at Epoch {}\n'.format(
                criterion,
                self.trlog[max_acc],
                self.trlog[max_acc_interval],
                 self.trlog[max_acc_epoch])
        print(max_validation_str)

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                embeddings, logits = self.model(data, return_feature=True)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc

                if args.tst_free:

                    embeddings_dict = self.model.get_embeddings_dict(embeddings, all_labels)

                    # TST-free part
                    clustering_losses = tst_free.clustering_loss(embeddings_dict, args.sinkhorn_reg, 'wasserstein',
                                                                 temperature=np.sqrt(args.temperature),
                                                                 normalize_by_dim=False,
                                                                 clustering_iterations=20, sinkhorn_iterations=20,
                                                                 sinkhorn_iterations_warmstart=4,
                                                                 sanity_check=False)

                    for key, val in clustering_losses.items():
                        metrics[key].append(val)

                if args.debug_fast:
                    print('Debug fast, breaking TEST after 1 mini-batch')
                    record = record[:1]
                    break

        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        metric_summaries = {key: compute_confidence_interval(val) for key, val in metrics.items()}

        self.trlog[test_acc] = va
        self.trlog[test_acc_interval] = vap
        self.trlog[test_loss] = vl

        summary_lines = []
        summary_lines.append(max_validation_str)
        summary_lines.append('test_SupervisedAcc {:.4f} + {:.4f} (ep{})'.format(
                self.trlog[test_acc],
                self.trlog[test_acc_interval],
                self.trlog[max_acc_epoch]))
        for key, (mean, std) in metric_summaries.items():
            summary_lines.append('test_{} {:.4f} + {:.4f} (ep{})'.format(key, mean, std, self.trlog[max_acc_epoch]))

        #self.print_metric_summaries(metric_summaries, prefix='\ttest_')
        #self.log_metric_summaries(metric_summaries, 0, prefix='test_')
        self.trlog['TST'] = metric_summaries

        summary_lines_str = '\n'.join(summary_lines)
        print('\n{}'.format(summary_lines_str))

        with open(osp.join(self.args.save_path, 'summary_max_{}.txt'.format(criterion)), 'w') as f:
            f.write(summary_lines_str)

    def final_record(self):
        pass
