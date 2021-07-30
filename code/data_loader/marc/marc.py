from ..common import (
    SentencePairExample,
    MultilingualRawDataset,
    RawDataset,
)
import itertools
import os
import json
from collections import OrderedDict, Counter
from ..data_configs import abbre2language


class MARCDataset(MultilingualRawDataset):
    def __init__(self):
        self.name = "marc"
        self.lang_abbres = ["de", "en", "es", "fr", "zh", "ja"]
        self.metrics = ["accuracy", "MAE"]
        self.label_list = [1, 2, 3, 4, 5]
        self.label2idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        self.num_labels = 5
        self.num_trn_examples = 200000
        self.contents = OrderedDict()
        self.create_contents()

    def get_labels(self):
        return self.label_list

    def get_language_data(self, language):
        return self.contents[language]

    def create_contents(self):
        marc_ = "./data/marc/"
        entries = []
        for lang in self.lang_abbres:
            for which_split, wsplit in (
                ("train", "trn"),
                ("dev", "val"),
                ("test", "tst"),
            ):
                if which_split == "train":
                    which_split = os.path.join("train", f"dataset_{lang}_train.json")
                elif which_split == "dev":
                    which_split = os.path.join("dev", f"dataset_{lang}_dev.json")
                elif which_split == "test":
                    which_split = os.path.join("test", f"dataset_{lang}_test.json")
                file_ = os.path.join(marc_, which_split)
                entries.extend(self.marc_parse(lang, file_, wsplit))
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

    def marc_parse(self, lang, input_file, which_split):
        sentence_egs = []
        language = abbre2language[lang]
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = json.loads(line.strip())
                label = int(line["stars"])
                assert line["language"] == lang
                assert label in self.get_labels(), f"{label}, {input_file}"
                category = line["product_category"].strip()
                title = line["review_title"].strip()
                text_a = line["review_body"].strip()
                text_b = f"{title} . {category}"

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
        return sentence_egs
