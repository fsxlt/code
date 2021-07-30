import os
import pickle
import itertools

from data_loader.pos import UDPOSDataset


class SampledUDPOSDataset(UDPOSDataset):
    def __init__(self, num_shots, group_index):
        super(SampledUDPOSDataset, self).__init__()
        # override name and trn_egs
        self.name = f"udpos-{num_shots}_shots-{group_index}_th"
        self.num_shots = num_shots
        self.group_index = group_index
        self.update_egs()

    def update_egs(self):
        sampled_panx_ = "./sampled_infos/sampled_data/udpos/"
        entries = []
        for lang in self.lang_abbres:
            if lang == "en":
                # we have en egs there, in case we want to sample kshots en
                # during adaptation
                continue
            where_ = os.path.join(
                sampled_panx_,
                f"{self.num_shots}-shots",
                lang,
                f"{self.group_index}-th",
            )
            assert os.path.exists(where_)
            file_ = os.path.join(where_, "train.tsv")
            entries.extend(self.udpos_parse(lang, file_, 'trn', update_label_list=False))

        entries = sorted(entries, key=lambda x: x[0])  # groupby requires contiguous
        for language, triplets in itertools.groupby(entries, key=lambda x: x[0]):
            # language, [(lang, split, eg)...]
            triplets = list(triplets)
            trn_egs = []
            for _language, split, eg in triplets:
                assert language == _language and split == "trn"
                trn_egs.append(eg)
            print(
                f"[INFO] {self.name} overrides trn_egs for {language}, len: {len(trn_egs)}"
            )
            self.contents[language].trn_egs = trn_egs
