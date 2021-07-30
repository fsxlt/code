import copy
import json



class RawDataset(object):
    def __init__(self, name, language, metrics, label_list, label2idx):
        self.name = name
        self.language = language
        self.metrics = metrics
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.label2idx = label2idx
        self.trn_egs = None
        self.val_egs = None
        self.tst_egs = None


class FSRawDataset(RawDataset):
    def __init__(self):
        raise NotImplementedError


class MultilingualRawDataset(object):
    def __init__(self, name, languages, metrics, label_list, label2idx):
        raise NotImplementedError
        self.name = name
        self.languages = languages
        self.metrics = metrics
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.label2idx = None
        self.contents = None

    def get_language_data(self, language):
        raise NotImplementedError


class SentenceExample(object):
    def __init__(self, uid, text_a, label=None, portion_identifier=-1):
        self.uid = uid
        self.text_a = text_a
        self.label = label
        self.portion_identifier = portion_identifier

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SentencePairExample(SentenceExample):
    def __init__(self, uid, text_a, text_b=None, label=None, portion_identifier=-1):
        super(SentencePairExample, self).__init__(
            uid, text_a, label, portion_identifier
        )
        self.text_b = text_b
