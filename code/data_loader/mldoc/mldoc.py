from ..common import (
    SentenceExample,
    MultilingualRawDataset,
    RawDataset,
)
import itertools
import os
from collections import OrderedDict
from ..data_configs import abbre2language


class MLDocDataset(MultilingualRawDataset):
    def __init__(self):
        self.name = "mldoc"
        self.lang_abbres = ["de", "en", "es", "fr", "it", "ru", "zh", "ja"]
        self.metrics = ["accuracy"]
        self.label_list = ["CCAT", "ECAT", "GCAT", "MCAT"]
        self.label2idx = {"CCAT": 0, "ECAT": 1, "GCAT": 2, "MCAT": 3}
        self.num_labels = 4
        self.num_trn_examples = 10000
        self.contents = OrderedDict()
        self.create_contents()

    def get_labels(self):
        return self.label_list

    def get_language_data(self, language):
        return self.contents[language]

    def create_contents(self):
        mldoc_ = "./data/mldoc/"
        entries = []
        for lang in self.lang_abbres:
            for which_split, wsplit in (
                ("train", "trn"),
                ("dev", "val"),
                ("test", "tst"),
            ):
                if which_split == "train":
                    which_split = f"{lang}.train.{self.num_trn_examples}"
                if which_split == "dev":
                    which_split = f"{lang}.dev"
                if which_split == "test":
                    which_split = f"{lang}.test"
                file_ = os.path.join(mldoc_, lang, which_split)
                entries.extend(self.mldoc_parse(lang, file_, wsplit))
        entries = sorted(entries, key=lambda x: x[0])  # groupby requires contiguous
        for language, triplets in itertools.groupby(entries, key=lambda x: x[0]):
            # language, [(lang, split, eg)...]
            triplets = list(triplets)
            trn_egs, val_egs, tst_egs = [], [], []
            for _language, split, eg in triplets:
                assert language == _language
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
                label2idx=self.label2idx,  # make all have the same mapping
            )
            _dataset.trn_egs = trn_egs if len(trn_egs) else None
            _dataset.val_egs = val_egs if len(val_egs) else None
            _dataset.tst_egs = tst_egs if len(tst_egs) else None

            self.contents[language] = _dataset

    def mldoc_parse(self, lang, input_file, which_split):
        sentence_egs = []
        language = abbre2language[lang]
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                if lang != "en":
                    assert len(line) == 2
                portion_identifier = -1
                label = line[0].strip()
                text_a = line[1].strip()
                assert label in self.get_labels(), f"{label}, {input_file}"
                sentence_egs.append(
                    (
                        language,
                        which_split,
                        SentenceExample(
                            uid=f"{language}-{idx}-{which_split}",
                            text_a=text_a,
                            label=label,
                            portion_identifier=portion_identifier,
                        ),
                    )
                )
        return sentence_egs
