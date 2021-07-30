from finetuning_parameters import get_args
from future.baseline_trainer import BaselineTuner
from future.modules import ptl2classes
from future.hooks import EvaluationRecorder

from data_loader.wrap_sampler import wrap_sampler
import data_loader.task_configs as task_configs
import data_loader.data_configs as data_configs
from future.collocate_fns import task2collocate_fn

import utils.checkpoint as checkpoint
import utils.logging as logging

import torch
import random
import os


config = dict(
    ptl="bert",
    model="bert-base-multilingual-cased",
    dataset_name="panx",
    experiment="debug",
    trn_languages="german",
    eval_languages=(
        "english,afrikaans,arabic,bulgarian,bengali,german,greek,spanish,"
        "estonian,basque,persian,finnish,french,hebrew,hindi,hungarian,"
        "indonesian,italian,japanese,javanese,georgian,kazakh,korean,"
        "malayalam,marathi,malay,burmese,dutch,portuguese,russian,"
        "swahili,tamil,telugu,thai,tagalog,turkish,urdu,vietnamese,yoruba,chinese"
    ),
    finetune_epochs=10,
    eval_every_batch=200,
    finetune_lr=7e-5,
    finetune_batch_size=32,
    inference_batch_size=512,
    world="0",
    train_fast=True,
    manual_seed=42,
    max_seq_len=128,
)


def init_task(conf):
    raw_dataset = task_configs.task2dataset[conf.dataset_name]()
    metric_name = raw_dataset.metrics[0]
    classes = ptl2classes[conf.ptl]
    tokenizer = classes.tokenizer.from_pretrained(conf.model)
    if conf.dataset_name in ["conll2003", "panx", "udpos"]:
        model = classes.seqtag.from_pretrained(
            conf.model, out_dim=raw_dataset.num_labels
        )
    elif conf.dataset_name in ["mldoc", "marc", "pawsx", "argustan", "xnli"]:
        model = classes.seqcls.from_pretrained(
            conf.model, num_labels=raw_dataset.num_labels
        )
    else:
        raise ValueError(f"{conf.dataset_name} is not covered!")
    exp_languages = sorted(list(set(conf.trn_languages + conf.eval_languages)))
    data_iter_cls = data_configs.task2dataiter[conf.dataset_name]
    data_iter = {}
    if hasattr(raw_dataset, "contents"):
        # multilingual dataset
        for language in exp_languages:
            data_iter[language] = data_iter_cls(
                raw_dataset=raw_dataset.contents[language],
                model=conf.model,
                tokenizer=tokenizer,
                max_seq_len=conf.max_seq_len,
            )
    else:
        data_iter[raw_dataset.language] = data_iter_cls(
            raw_dataset=raw_dataset,
            model=conf.model,
            tokenizer=tokenizer,
            max_seq_len=conf.max_seq_len,
        )
    collocate_batch_fn = task2collocate_fn[conf.dataset_name]
    return (model, tokenizer, data_iter, metric_name, collocate_batch_fn)


def init_hooks(conf, metric_name):
    eval_recorder = EvaluationRecorder(
        where_=os.path.join(conf.checkpoint_root, "state_dicts"), which_metric=metric_name
    )
    return [eval_recorder]


def main(conf):

    if conf.override:
        for name, value in config.items():
            assert type(getattr(conf, name)) == type(value), f"{name} {value}"
            setattr(conf, name, value)

    init_config(conf)

    # init model
    model, tokenizer, data_iter, metric_name, collocate_batch_fn = init_task(conf)
    adapt_loaders = {}
    for language, language_dataset in data_iter.items():
        # NOTE: the sample dataset are refered
        adapt_loaders[language] = wrap_sampler(
            trn_batch_size=conf.finetune_batch_size,
            infer_batch_size=conf.inference_batch_size,
            language=language,
            language_dataset=language_dataset,
        )

    hooks = init_hooks(conf, metric_name)

    conf.logger.log("Initialized tasks, recorders, and initing the trainer.")
    trainer = BaselineTuner(
        conf, collocate_batch_fn=collocate_batch_fn, logger=conf.logger
    )

    conf.logger.log("Starting training/validation.")
    trainer.train(
        model,
        tokenizer=tokenizer,
        data_iter=data_iter,
        metric_name=metric_name,
        adapt_loaders=adapt_loaders,
        hooks=hooks,
    )

    # update the status.
    conf.logger.log("Finishing training/validation.")
    conf.is_finished = True
    logging.save_arguments(conf)


def init_config(conf):
    conf.is_finished = False
    conf.task = conf.dataset_name

    # device
    assert conf.world is not None, "Please specify the gpu ids."
    conf.world = (
        [int(x) for x in conf.world.split(",")]
        if "," in conf.world
        else [int(conf.world)]
    )
    conf.n_sub_process = len(conf.world)

    # re-configure batch_size if sub_process > 1.
    if conf.n_sub_process > 1:
        conf.finetune_batch_size = conf.finetune_batch_size * conf.n_sub_process

    conf.trn_languages = (
        [x for x in conf.trn_languages.split(",")]
        if "," in conf.trn_languages
        else [conf.trn_languages]
    )

    conf.eval_languages = (
        [x for x in conf.eval_languages.split(",")]
        if "," in conf.eval_languages
        else [conf.eval_languages]
    )

    random.seed(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)
    torch.cuda.manual_seed(conf.manual_seed)

    assert torch.cuda.is_available()
    torch.cuda.set_device(conf.world[0])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # define checkpoint for logging.
    checkpoint.init_checkpoint_baseline(conf)

    # display the arguments' info.
    logging.display_args(conf)

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_root)


if __name__ == "__main__":
    parser = get_args()
    # parse conf.
    conf = parser.parse_args()

    main(conf)
