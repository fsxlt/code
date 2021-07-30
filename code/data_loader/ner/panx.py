from ..common import RawDataset, MultilingualRawDataset
from ..data_configs import task2datadir, abbre2language
import itertools
from collections import OrderedDict, defaultdict
import os


class PANXDataset(MultilingualRawDataset):
    def __init__(self):
        self.name = "panx"
        self.data_dir = task2datadir[self.name]
        # ("english,afrikaans,arabic,bulgarian,bengali,german,greek,spanish,"
        # "estonian,basque,persian,finnish,french,hebrew,hindi,hungarian,"
        # "indonesian,italian,japanese,javanese,georgian,kazakh,korean,"
        # "malayalam,marathi,malay,burmese,dutch,portuguese,russian,"
        # "swahili,tamil,telugu,thai,tagalog,turkish,urdu,vietnamese,yoruba,chinese")
        self.lang_abbres = [
            "en",
            "af",
            "ar",
            "bg",
            "bn",
            "de",
            "el",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fr",
            "he",
            "hi",
            "hu",
            "id",
            "it",
            "ja",
            "jv",
            "ka",
            "kk",
            "ko",
            "ml",
            "mr",
            "ms",
            "my",
            "nl",
            "pt",
            "ru",
            "sw",
            "ta",
            "te",
            "th",
            "tl",
            "tr",
            "ur",
            "vi",
            "yo",
            "zh",
        ]
        self.metrics = ["f1_score_tagging"]
        self.label_list = []
        self.label2idx = {}
        self.pad_id = None
        self.num_labels = -1
        self.contents = OrderedDict()
        self.create_contents()

    def get_labels(self):
        return self.label_list

    def get_language_data(self, language):
        return self.contents[language]

    def create_contents(self):
        entries = []
        for lang in self.lang_abbres:
            for which_split, wsplit in (
                ("train", "trn"),
                ("dev", "val"),
                ("test", "tst"),
            ):
                if which_split == "train":
                    which_split = f"train-{lang}.tsv"
                if which_split == "dev":
                    which_split = f"dev-{lang}.tsv"
                if which_split == "test":
                    which_split = f"test-{lang}.tsv"
                file_ = os.path.join(self.data_dir, which_split)
                entries.extend(self.panx_parse(lang, file_, wsplit))
        entries = sorted(entries, key=lambda x: x[0])  # groupby requires contiguous
        self.label_list = sorted(list(set(self.label_list)))
        self.label_list.append("<PAD>")
        self.label2idx = {t: i for i, t in enumerate(self.label_list)}
        self.pad_id = self.label2idx["<PAD>"]
        self.num_labels = len(self.label_list)
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
            # only for tagging task
            _dataset.idx2label = {v: k for k, v in _dataset.label2idx.items()}

            self.contents[language] = _dataset

    def panx_parse(self, lang, input_file, which_split, update_label_list=True):
        # or just use nltk
        sentence_egs = []
        language = abbre2language[lang]
        with open(input_file, "r") as f:
            lines = f.read().strip().split("\n\n")
            for line in lines:
                sent_vec = line.strip().split("\n")
                token_tag_vec = [wt.strip().split("\t") for wt in sent_vec]
                if update_label_list:
                    for _, tag in token_tag_vec:
                        self.label_list.append(tag)
                sentence_egs.append((language, which_split, token_tag_vec,))
        return sentence_egs
