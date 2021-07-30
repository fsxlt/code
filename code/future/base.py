from torch.utils.data import SequentialSampler, RandomSampler
from .hooks import EvaluationRecorder
import utils.eval_meters as eval_meters
from seqeval.metrics import f1_score as f1_score_tagging
import torch


class BaseTrainer(object):
    def __init__(self, conf, logger, criterion=torch.nn.CrossEntropyLoss()):
        self.conf = conf
        self.logger = logger
        self.log_fn_json = logger.log_metric
        self.log_fn = logger.log
        self.criterion = criterion

        self._batch_step = 0
        self._epoch_step = 0

    @property
    def batch_step(self):
        return self._batch_step

    @property
    def epoch_step(self):
        return self._epoch_step

    def _parallel_to_device(self, model):
        model = model.cuda()
        if len(self.conf.world) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.conf.world)
        return model

    def _model_forward(self, model, **kwargs):
        if self.model_ptl == "distilbert" and "token_type_ids" in kwargs:
            kwargs.pop("token_type_ids")
        return model(**kwargs)

    def _infer_one_loader(
        self, model, loader, collocate_batch_fn, metric_name="accuracy", device=None
    ):
        assert isinstance(loader.sampler, SequentialSampler)
        try:
            eval_fn = getattr(eval_meters, metric_name)
        except:
            raise ValueError(
                f"Required metric {metric_name} not implemented in meters module."
            )
        if device is None:
            device = torch.cuda.current_device()
        model.eval()
        all_golds, all_preds = [], []
        for batched in loader:
            batched, golds, *_ = collocate_batch_fn(batched, device=device)
            with torch.no_grad():
                logits, *_ = self._model_forward(model, **batched)
                preds = torch.argmax(logits, dim=-1)
            all_golds.extend(golds.tolist())
            all_preds.extend(preds.tolist())
        assert len(all_golds) == len(all_preds)
        eval_res = eval_fn(all_golds, all_preds)
        model.train()
        return eval_res, metric_name

    def _infer_one_loader_tagging(
        self,
        model,
        idx2label,
        loader,
        collocate_batch_fn,
        metric_name="f1_score_tagging",
        device=None,
    ):
        if device is None:
            device = torch.cuda.current_device()
        model.eval()
        all_preds_tagging, all_golds_tagging = [], []
        for batched in loader:
            batched, golds, uids, _golds_tagging = collocate_batch_fn(
                batched, device=device
            )
            with torch.no_grad():
                _, bert_out_preds, *_ = self._model_forward(model, **batched)
                assert bert_out_preds.shape == _golds_tagging.shape
                if_tgts = batched["if_tgts"]
                for sent_idx in range(_golds_tagging.shape[0]):
                    sent_gold = _golds_tagging[sent_idx][if_tgts[sent_idx]]
                    sent_pred = bert_out_preds[sent_idx][if_tgts[sent_idx]]
                    all_golds_tagging.append(
                        [idx2label[label_id.item()] for label_id in sent_gold]
                    )
                    all_preds_tagging.append(
                        [idx2label[label_id.item()] for label_id in sent_pred]
                    )
        assert len(all_golds_tagging) == len(all_preds_tagging)
        eval_fn = eval(metric_name)
        eval_res = eval_fn(all_preds_tagging, all_golds_tagging)
        model.train()
        return eval_res, metric_name

    @staticmethod
    def _get_eval_recorder_hook(hook_container):
        for hook in hook_container.hooks:
            if isinstance(hook, EvaluationRecorder):
                return hook
