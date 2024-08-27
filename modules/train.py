import torch
from matplotlib.colors import ListedColormap
from torch import optim
from tqdm import tqdm
import random
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup
import csv
from .metrics import eval_result
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()


class RETrainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, args=None,
                 logger=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.re_dict = processor.get_relation_dict()
        self.logger = logger
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.optimizer = None
        self.scheduler = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args
        if self.args.use_prompt:
            self.before_multimodal_train()
        else:
            self.before_train()

    def train(self):
        self.step = 0
        self.model[0].train()
        self.model[1].train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model[0].load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    outputs, labels = self._step(batch, mode="train")
                    loss = outputs[0]

                    logits_homo, reprs_homo = [], []
                    reprs_homo.append(outputs[1])
                    reprs_homo.append(outputs[2])
                    logits_homo.append(outputs[3])
                    logits_homo.append(outputs[4])
                    logits_homo = torch.stack(logits_homo)
                    reprs_homo = torch.stack(reprs_homo)
                    edges_homo, edges_origin_homo = self.model[1](logits_homo, reprs_homo)
                    loss_reg_homo, loss_logit_homo, loss_repr_homo = \
                        self.model[1].distillation_loss(logits_homo, reprs_homo, edges_homo)
                    graph_distill_loss_homo = 0.05 * (loss_logit_homo + loss_reg_homo)
                    loss = loss + graph_distill_loss_homo

                    loss = loss.mean()
                    avg_loss += loss.detach().cpu().item()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss,
                                                   global_step=self.step)  # tensorbordx
                        avg_loss = 0

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)  # generator to dev.

            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch,
                                                                                                    self.best_dev_metric))
            self.logger.info(
                "Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch,
                                                                                         self.best_test_metric))

    def evaluate(self, epoch):
        self.model[0].eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        step = 0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                # 遍历dev_data
                # batch: (input_ids, token_type_ids, attention_mask, labels)
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                             batch)  # to cpu/cuda device
                    outputs, labels = self._step(batch, mode="dev")
                    loss = outputs[0]
                    logits = outputs[5]
                    loss = loss.mean()
                    total_loss += loss.detach().cpu().item()
                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels,
                                                     labels=list(self.re_dict.values())[1:],
                                                     target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)
                if self.writer:
                    self.writer.add_scalar(tag='dev_acc', scalar_value=acc, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='dev_f1', scalar_value=micro_f1, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss / len(self.test_data),
                                           global_step=epoch)  # tensorbordx

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch,
                                         micro_f1, acc))
                if micro_f1 >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1  # update best metric(f1 score)
                    if self.args.save_path is not None:
                        torch.save(self.model[0].state_dict(), self.args.save_path + "/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model[0].train()

    def test(self):
        self.model[0].eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        # self.tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased', do_lower_case=True)
        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model[0].load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        true_labels, pred_labels = [], []

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                all_sample_vectors = np.empty((0, 768))
                all_text_features = np.empty((0, 1536))
                all_visual_features = np.empty((0, 768))
                all_labels = np.empty((0, 1))

                for batch in self.test_data:
                    # batch_list = batch

                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                             batch)  # to cpu/cuda device
                    outputs, labels = self._step(batch, mode="dev")
                    loss = outputs[0]
                    logits = outputs[1]
                    # x = outputs[2]
                    uni_l = outputs[2]
                    uni_v = outputs[3]

                    loss = loss.mean()
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())

                    pbar.update()

                # evaluate done
                pbar.close()
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels,
                                                     labels=list(self.re_dict.values())[1:],
                                                     target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)
                if self.writer:
                    self.writer.add_scalar(tag='test_acc', scalar_value=acc)  # tensorbordx
                    self.writer.add_scalar(tag='test_f1', scalar_value=micro_f1)  # tensorbordx
                    self.writer.add_scalar(tag='test_loss',
                                           scalar_value=total_loss / len(self.test_data))  # tensorbordx
                total_loss = 0
                self.logger.info("Test f1 score: {}, acc: {}.".format(micro_f1, acc))
        self.model[0].train()

    def _step(self, batch, mode="train"):
        if mode != "predict":
            if self.args.use_prompt:
                input_ids, token_type_ids, attention_mask, input_ids_phrase, attention_mask_phrase, labels, images, aux_imgs = batch
            else:
                images, aux_imgs = None, None
                input_ids, token_type_ids, attention_mask, labels = batch
            outputs = self.model[0](input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                 input_ids_phrase=input_ids_phrase, attention_mask_phrase=attention_mask_phrase,
                                 labels=labels, images=images, aux_imgs=aux_imgs)
            return outputs, labels

    def before_train(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model[0].named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in self.model[0].named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model[0].to(self.args.device)

    def before_multimodal_train(self):
        optimizer_grouped_parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model[0].named_parameters():
            if 'bert' in name:
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)

        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model[0].named_parameters():
            if 'encoder_conv' in name or 'gates' in name:
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)

        # freeze resnet
        for name, param in self.model[0].named_parameters():
            if 'image_model' in name:
                param.require_grad = False
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model[0].to(self.args.device)






