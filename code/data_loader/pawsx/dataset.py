from ..common import (
    SentencePairExample,
    MultilingualRawDataset,
    RawDataset,
)
import itertools
import os
import json
from collections import OrderedDict
from ..data_configs import abbre2language


class PAWSXDataset(MultilingualRawDataset):
    def __init__(self):
        self.name = "pawsx"
        self.lang_abbres = ["de", "en", "es", "fr", "ja", "ko", "zh"]
        self.metrics = ["accuracy"]
        self.label_list = ["0", "1"]
        self.label2idx = {"0": 0, "1": 1}
        self.num_labels = 2
        self.contents = OrderedDict()
        self.create_contents()

    def get_labels(self):
        return self.label_list

    def get_language_data(self, language):
        return self.contents[language]

    def create_contents(self):
        pawsx_ = "./data/pawsx/"
        entries = []
        for lang in self.lang_abbres:
            for which_split, wsplit in (
                ("train", "trn"),
                ("dev", "val"),
                ("test", "tst"),
            ):
                if which_split == "train":
                    which_split = f"train.tsv"
                elif which_split == "dev":
                    which_split = f"dev_2k.tsv"
                elif which_split == "test":
                    which_split = f"test_2k.tsv"
                file_ = os.path.join(pawsx_, lang, which_split)
                if not os.path.exists(file_):
                    print(f"[INFO]: skip {lang} {wsplit}: not such file")
                    continue
                entries.extend(self.pawsx_parse(lang, file_, wsplit))
        entries = sorted(entries, key=lambda x: x[0])  # groupby requires contiguous
        for language, triplets in itertools.groupby(entries, key=lambda x: x[0]):
            # get examples in this language
            triplets = list(triplets)
            trn_egs, val_egs, tst_egs = [], [], []
            for _, split, eg in triplets:
                if split == "trn":
                    trn_egs.append(eg)
                elif split == "val":
                    val_egs.append(eg)
                elif split == "tst":
                    tst_egs.append(eg)
                else:
                    raise ValueError
            _dataset = RawDataset(
                name=f"{self.name}-{language}",
                language=language,
                metrics=self.metrics,
                label_list=self.label_list,
                label2idx=self.label2idx,
            )
            _dataset.trn_egs = trn_egs if len(trn_egs) else None
            _dataset.val_egs = val_egs if len(val_egs) else None
            _dataset.tst_egs = tst_egs if len(tst_egs) else None

            self.contents[language] = _dataset

    def pawsx_parse(self, lang, input_file, which_split):
        sentence_egs = []
        language = abbre2language[lang]
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                line = line.strip().split("\t")
                assert len(line) == 4
                text_a, text_b, label = line[1], line[2], line[-1]
                assert label in self.get_labels(), f"{label}, {input_file}"
                portion_identifier = -1
                sentence_egs.append(
                    (
                        language,
                        which_split,
                        SentencePairExample(
                            uid=f"{language}-{idx}-{which_split}",
                            text_a=text_a,
                            text_b=text_b,
                            label=label,
                            portion_identifier=portion_identifier,
                        ),
                    )
                )
        print(input_file, len(sentence_egs))
        return sentence_egs
