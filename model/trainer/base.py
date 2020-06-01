import abc
import torch
import os.path as osp

from model.utils import (
    ensure_path,
    Averager, Timer, count_acc,
    compute_confidence_interval,
)
from model.logger import Logger

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        # ensure_path(
        #     self.args.save_path,
        #     scripts_to_save=['model/models', 'model/networks', __file__],
        # )
        self.logger = Logger(args, osp.join(args.save_path))

        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc_interval'] = 0.0

        # For tst
        if args.tst_free:
            self.trlog['max_tst_criterion'] = 0.0
            self.trlog['max_tst_criterion_interval'] = 0.
            self.trlog['max_tst_criterion_epoch'] = 0
            self.trlog['tst_criterion'] = args.tst_criterion

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, data_loader):
        pass
    
    @abc.abstractmethod
    def evaluate_test(self, data_loader):
        pass    
    
    @abc.abstractmethod
    def final_record(self):
        pass    

    def print_metric_summaries(self, metric_summaries, prefix='\t'):
        for key, (mean, std) in metric_summaries.items():
            print('{}{}: {:.4f} +/- {:.4f}'.format(prefix, key, mean, std))

    def log_metric_summaries(self, metric_summaries, epoch, prefix=''):
        for key, (mean, std) in metric_summaries.items():
            self.logger.add_scalar('{}{}'.format(prefix, key), mean, epoch)

    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch % args.eval_interval == 0:

            if not args.tst_free:
                vl, va, vap = self.evaluate(self.val_loader)
                self.logger.add_scalar('val_loss', float(vl), self.train_epoch)
                self.logger.add_scalar('val_acc', float(va),  self.train_epoch)
                print('epoch {}, val, loss={:.4f} acc={:.4f}+{:.4f}'.format(epoch, vl, va, vap))
            else:
                vl, va, vap, metrics = self.evaluate(self.val_loader)
                self.logger.add_scalar('val_loss', float(vl), self.train_epoch)
                self.logger.add_scalar('val_acc', float(va),  self.train_epoch)
                print('epoch {}, val, loss={:.4f} acc={:.4f}+{:.4f}'.format(epoch, vl, va, vap))
                self.print_metric_summaries(metrics, prefix='\tval_')
                self.log_metric_summaries(metrics, epoch=epoch, prefix='val_')

            if va >= self.trlog['max_acc']:
                self.trlog['max_acc'] = va
                self.trlog['max_acc_interval'] = vap
                self.trlog['max_acc_epoch'] = self.train_epoch
                self.save_model('max_acc')

            # Probably a different criterion for TST -> optimize here.
            if args.tst_free and args.tst_criterion:
                assert args.tst_criterion in metrics, 'Criterion {} not found in {}'.format(args.tst_criterion, metrics.keys())
                criterion, criterion_interval = metrics[args.tst_criterion]
                if criterion >= self.trlog['max_tst_criterion']:
                    self.trlog['max_tst_criterion'] = criterion
                    self.trlog['max_tst_criterion_interval'] = criterion_interval
                    self.trlog['max_tst_criterion_epoch'] = self.train_epoch
                    self.save_model('max_tst_criterion')
                    print('Found new best model at Epoch {} : Validation {} = {:.4f} +/- {:4f}'.format(
                        self.train_epoch, args.tst_criterion, criterion, criterion_interval))


    def try_logging(self, tl1, tl2, ta, tg=None):
        args = self.args
        if self.train_step % args.log_interval == 0:
            print('epoch {}, train {:06g}/{:06g}, total loss={:.4f}, loss={:.4f} acc={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.train_step,
                          self.max_steps,
                          tl1.item(), tl2.item(), ta.item(),
                          self.optimizer.param_groups[0]['lr']))
            self.logger.add_scalar('train_total_loss', tl1.item(), self.train_step)
            self.logger.add_scalar('train_loss', tl2.item(), self.train_step)
            self.logger.add_scalar('train_acc',  ta.item(), self.train_step)
            if tg is not None:
                self.logger.add_scalar('grad_norm',  tg.item(), self.train_step)
            print('data_timer: {:.2f} sec, '     \
                  'forward_timer: {:.2f} sec,'   \
                  'backward_timer: {:.2f} sec, ' \
                  'optim_timer: {:.2f} sec'.format(
                        self.dt.item(), self.ft.item(),
                        self.bt.item(), self.ot.item())
                  )
            self.logger.dump()

    def save_model(self, name):
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
