import json
import copy
import torch
from tqdm import tqdm

import transformers, functools


def _patched_encode_plus(tokenizer):
    # monkey patch to pass transformers version mismatch
    if transformers.__version__ == "2.2.2":
        return tokenizer.encode_plus
    return functools.partial(tokenizer.encode_plus, truncation=True)


class BertInputFeature(object):
    def __init__(self, uid, input_ids, attention_mask, token_type_ids, label):
        self.uid = uid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.gold = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def glue_example_to_feature(
    task,
    examples,
    tokenizer,
    max_seq_len,
    label_list,
    pad_token=0,
    pad_token_segment_id=0,
):

    assert pad_token == pad_token_segment_id == 0
    print(
        "[INFO]: using following label set for task {} : {}.".format(task, label_list)
    )
    patched_encode_plus = _patched_encode_plus(tokenizer)
    # associate each label with an index
    label_map = {l: i for i, l in enumerate(label_list)}

    features = []
    print("[INFO] *** Convert Example to Features ***")
    for idx, eg in enumerate(tqdm(examples)):
        if not hasattr(eg, "text_b"):  
            setattr(eg, "text_b", None)
        inputs = patched_encode_plus(
            eg.text_a, eg.text_b, add_special_tokens=True, max_length=max_seq_len
        )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        # pad everything to max_seq_len
        padding_len = max_seq_len - len(input_ids)
        input_ids = input_ids + [pad_token] * padding_len
        attention_mask = attention_mask + [0] * padding_len
        token_type_ids = token_type_ids + [pad_token_segment_id] * padding_len
        assert (
            len(input_ids) == len(attention_mask) == len(token_type_ids) == max_seq_len
        ), "{} - {} - {}".format(
            len(input_ids), len(attention_mask), len(token_type_ids)
        )

        if idx < 2:
            print()
            print("[DEBUG] *** Example Entries in Dataset ***")
            print("[DEBUG] uid: {}".format(eg.uid))
            print("[DEBUG] input_ids: {}".format(" ".join([str(x) for x in input_ids])))
            print(
                "[DEBUG] attention_mask: {}".format(
                    " ".join([str(x) for x in attention_mask])
                )
            )
            print(
                "[DEBUG] token_type_ids: {}".format(
                    " ".join([str(x) for x in token_type_ids])
                )
            )

        features.append(
            BertInputFeature(
                uid=eg.uid,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label_map[eg.label],
            )
        )
    return features


""" for POS tagging on PTB, I use the last wordpiece to represent the word """

# following tags are not considered in tagging on PTB

_skipped_tags = {"-NONE-", "NFP", "AFX"}


class TaggingBertInputFeature(BertInputFeature):
    def __init__(self, uid, input_ids, attention_mask, sent_if_tgt, tags_ids):
        super(TaggingBertInputFeature, self).__init__(
            uid, input_ids, attention_mask, None, None
        )
        self.sent_if_tgt = sent_if_tgt
        self.tags_ids = tags_ids


def tagging_example_to_feature(which_split, tagged_sents, tokenizer, t2i, msl):

    all_fts, toolongs = [], []
    for sent_idx, sent in enumerate(tqdm(tagged_sents)):
        sent_pieces, sent_piece_tags, sent_if_tgt = [], [], []
        for word, tag in sent:
            word_pieces = tokenizer.tokenize(word)
            piece_tags = ["<PAD>"] * (len(word_pieces) - 1) + [tag]
            if tag in _skipped_tags:
                piece_if_tgt = [0] * (len(word_pieces) - 1) + [0]
            else:
                piece_if_tgt = [0] * (len(word_pieces) - 1) + [1]
            sent_pieces.extend(word_pieces)
            sent_piece_tags.extend(piece_tags)
            sent_if_tgt.extend(piece_if_tgt)
        if len(sent_pieces) > msl - 2:
            # print(sent_pieces)
            print("{} > {} in {} ...".format(len(sent_pieces), msl - 2, which_split))
            toolongs.append(len(sent_pieces))
            sent_pieces, sent_piece_tags, sent_if_tgt = map(
                lambda x: x[: (msl - 2)], [sent_pieces, sent_piece_tags, sent_if_tgt],
            )
        sent_pieces = ["[CLS]"] + sent_pieces + ["[SEP]"]
        sent_piece_tags = ["<PAD>"] + sent_piece_tags + ["<PAD>"]
        sent_if_tgt = [0] + sent_if_tgt + [0]
        bert_inp_ids = tokenizer.convert_tokens_to_ids(sent_pieces)
        bert_inp_mask = [1] * len(bert_inp_ids)
        tags_ids = [t2i[tag] for tag in sent_piece_tags]

        assert len(sent_pieces) == len(sent_if_tgt) == len(tags_ids)

        while len(bert_inp_ids) < msl:
            bert_inp_ids.append(0)
            bert_inp_mask.append(0)
            sent_if_tgt.append(0)
            tags_ids.append(t2i["<PAD>"])

        all_fts.append(
            TaggingBertInputFeature(
                uid="{}-{}".format(which_split, sent_idx),
                input_ids=bert_inp_ids,
                attention_mask=bert_inp_mask,
                sent_if_tgt=sent_if_tgt,
                tags_ids=tags_ids,
            )
        )
    print("[WARN]: {} sentences longer than msl ...".format(len(toolongs)))
    return all_fts
