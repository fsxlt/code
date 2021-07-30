# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from .base import BaseTrainer
from .hooks.base_hook import HookContainer
from .hooks import EvaluationRecorder
from torch.utils.data import SequentialSampler, RandomSampler
from collections import defaultdict, Counter


class AdaptTuner(BaseTrainer):
    def __init__(self, conf, collocate_batch_fn, logger):
        assert len(conf.adapt_trn_languages) == 1
        super(AdaptTuner, self).__init__(conf, logger)
        self.log_fn("Init trainer.")
        self.collocate_batch_fn = collocate_batch_fn
        self.model_ptl = conf.ptl

    def _init_model_opt(self, model):
        model = self._parallel_to_device(model)
        trn_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trn_params, lr=self.conf.adapt_lr)
        opt.zero_grad()
        model.zero_grad()
        return opt, model

    def _infer_tst_egs(
        self, hook_container, data_iter, metric_name, adapt_loaders, tst_languages
    ):
        assert isinstance(tst_languages, list)
        best_model = deepcopy(
            self._get_eval_recorder_hook(hook_container).best_state["best_state_dict"]
        ).cuda()
        scores = defaultdict(dict)
        for language in tst_languages:
            for split_name in ["tst_egs"]:
                loader = getattr(adapt_loaders[language], split_name)
                if self.conf.dataset_name in ["conll2003", "panx", "udpos"]:
                    eval_res, *_ = self._infer_one_loader_tagging(
                        model=best_model,
                        idx2label=data_iter[language].raw_dataset.idx2label,
                        loader=loader,
                        collocate_batch_fn=self.collocate_batch_fn,
                        metric_name=metric_name,
                    )
                else:
                    eval_res, *_ = self._infer_one_loader(
                        model=best_model,
                        loader=loader,
                        collocate_batch_fn=self.collocate_batch_fn,
                        metric_name=metric_name,
                    )
                scores[language][split_name] = eval_res
        return scores

    def train(
        self, model, tokenizer, data_iter, metric_name, adapt_loaders, hooks=None
    ):
        opt, model = self._init_model_opt(model)
        self.model = model
        self.model.train()

        hook_container = HookContainer(world_env={"trainer": self}, hooks=hooks)
        hook_container.on_train_begin()

        adapt_language = self.conf.adapt_trn_languages[0]
        learning_curves = {"val_egs": defaultdict(list)}

        for epoch_index in range(1, self.conf.adapt_epochs + 1):
            all_uids, epoch_losses = [], []
            for batched in adapt_loaders[adapt_language].trn_egs:
                batched, golds, uids, _golds_tagging = self.collocate_batch_fn(batched)
                logits, *_ = self._model_forward(self.model, **batched)
                loss = self.criterion(logits, golds).mean()
                epoch_losses.append(loss.item())
                loss.backward()
                opt.step()
                opt.zero_grad()
                all_uids.extend(uids)
                self._batch_step += 1
            epoch_losses_str = "->".join(
                [f"{epoch_loss:.3f}" for epoch_loss in epoch_losses]
            )
            self.log_fn(
                f"epoch loss: {np.mean(epoch_losses):.3f} "
                f"epoch @ {epoch_index} "
                f"detailed loss {epoch_losses_str}"
            )
            self.log_fn(f"{all_uids}")
            self.log_fn("*" * 10)

            scores = defaultdict(dict)
            for language in [adapt_language]:
                for split_name in ["val_egs"]:
                    loader = getattr(adapt_loaders[language], split_name)
                    if self.conf.dataset_name in ["conll2003", "panx", "udpos"]:
                        eval_res, *_ = self._infer_one_loader_tagging(
                            self.model,
                            data_iter[language].raw_dataset.idx2label,
                            loader,
                            self.collocate_batch_fn,
                        )
                    else:
                        eval_res, *_ = self._infer_one_loader(
                            model=self.model,
                            loader=loader,
                            collocate_batch_fn=self.collocate_batch_fn,
                            metric_name=metric_name,
                        )
                    scores[language][split_name] = eval_res
                    learning_curves[split_name][language].append(eval_res)
            eval_score = scores[adapt_language]["val_egs"]
            hook_container.on_validation_end(eval_score=eval_score, all_scores=scores)
            best_epoch_step = self._get_eval_recorder_hook(hook_container).best_epoch
            if (
                self.conf.early_stop
                and epoch_index - best_epoch_step > self.conf.early_stop_patience
            ):
                self.log_fn(
                    f"Early-stopping: current epoch={epoch_index},"
                    f" best_epoch={best_epoch_step + 1}."
                )
                tst_scores = self._infer_tst_egs(
                    hook_container,
                    data_iter,
                    metric_name,
                    adapt_loaders,
                    [self.conf.adapt_trn_languages[0]],
                )
                hook_container.on_train_end(
                    learning_curves=learning_curves, tst_scores=tst_scores,
                )
                return
            self._epoch_step += 1
        tst_scores = self._infer_tst_egs(
            hook_container,
            data_iter,
            metric_name,
            adapt_loaders,
            [self.conf.adapt_trn_languages[0]],
        )
        hook_container.on_train_end(
            learning_curves=learning_curves, tst_scores=tst_scores,
        )
        return
