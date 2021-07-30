import os
import pickle

from data_loader.data_configs import abbre2language
from data_loader.mldoc import MLDocDataset


class SampledMLDocDataset(MLDocDataset):
    def __init__(self, num_shots, group_index):
        super(SampledMLDocDataset, self).__init__()
        # override name and trn_egs
        self.name = f"mldoc-{num_shots}_shots-{group_index}_th"
        self.num_shots = num_shots
        self.group_index = group_index
        self.update_egs()

    def update_egs(self):
        sampled_mldoc_ = "./sampled_infos/sampled_data/mldoc/"
        for lang in self.lang_abbres:
            if lang == "en":
                # we have en egs there, in case we want to sample kshots en
                # during adaptation
                continue
            language = abbre2language[lang]
            where_ = os.path.join(
                sampled_mldoc_,
                f"{self.num_shots}-shots",
                language,
                f"{self.group_index}-th",
            )
            data_ = [d_ for d_ in os.listdir(where_) if ".pkl" in d_]
            assert len(data_) == 1
            with open(os.path.join(where_, data_[0]), "rb") as f:
                kshots_buckets = pickle.load(f)
            trn_egs = []
            for gold_label, egs in kshots_buckets["actual_egs"].items():
                egs = [eg for eg in egs if eg is not None]
                for eg in egs:
                    assert language in eg.uid
                trn_egs.extend(egs)
            print(
                f"[INFO] {self.name} overrides trn_egs for {language}, len: {len(trn_egs)}"
            )
            self.contents[language].trn_egs = trn_egs
