import os
import pickle

from data_loader.data_configs import abbre2language
from data_loader.pawsx import PAWSXDataset


class SampledPAWSXDataset(PAWSXDataset):
    def __init__(self, num_shots, group_index):
        super(SampledPAWSXDataset, self).__init__()
        # override name and trn_egs
        self.name = f"pawsx-{num_shots}_shots-{group_index}_th"
        self.num_shots = num_shots
        self.group_index = group_index
        self.update_egs()

    def update_egs(self):
        sampled_pawsx = "./sampled_infos/sampled_data/pawsx/"
        for lang in self.lang_abbres:
            if lang == "en":
                continue
            language = abbre2language[lang]
            where_ = os.path.join(
                sampled_pawsx,
                f"{self.num_shots}-shots",
                language,
                f"{self.group_index}-th",
            )
            # get kshot trn_egs for this language
            data_ = [d_ for d_ in os.listdir(where_) if ".pkl" in d_]
            assert len(data_) == 1
            with open(os.path.join(where_, data_[0]), "rb") as f:
                kshots_buckets = pickle.load(f)
            trn_egs = []
            for gold_label, egs in kshots_buckets["actual_egs"].items():
                for eg in egs:
                    assert language in eg.uid
                trn_egs.extend(egs)
            print(
                f"[INFO] {self.name} overrides trn_egs for {language}, len: {len(trn_egs)}"
            )
            assert self.contents[language].trn_egs is None
            self.contents[language].trn_egs = trn_egs

            # get the common dev egs (holds for all groups) for this language
            val_where_ = os.path.join(
                sampled_pawsx, f"{self.num_shots}-shots", language, "left,dev.pkl"
            )
            val_egs = []
            with open(val_where_, "rb") as f:
                left_dev = pickle.load(f)
                for gold_label, egs in left_dev.items():
                    print(gold_label, len(egs))
                    val_egs.extend(egs)
            self.contents[language].val_egs = val_egs
