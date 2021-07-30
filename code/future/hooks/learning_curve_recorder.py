from .base_hook import Hook
import pickle
import os
import json


class LearningCurveRecorder(Hook):
    def __init__(self, where_):
        super(LearningCurveRecorder, self).__init__()
        if not os.path.isdir(where_):
            os.makedirs(where_)
        self.where_ = where_
        self.learning_where_ = os.path.join(where_, "learning_curves.json")
        self.name = self.__class__.__name__

    def on_train_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_validation_end(self):
        pass

    def on_train_end(self, learning_curves):
        """  learning_curves: {"val_egs": {"language": list}} """
        self.log_fn(f"learning_curves: {learning_curves}")
        with open(self.learning_where_, "w") as f:
            json.dump(learning_curves, f)
