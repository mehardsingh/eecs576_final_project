import pandas as pd
import os

class MetricHandler:
    def __init__(self, save_dir):
        self.batch_metric_dict = dict()
        self.metric_dict = dict()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def batch_update(self, single_batch_metric_dict):
        for metric_name in single_batch_metric_dict:
            if not metric_name in self.batch_metric_dict:
                self.batch_metric_dict[metric_name] = list()
                self.metric_dict[metric_name] = list()

            self.batch_metric_dict[metric_name].append(single_batch_metric_dict[metric_name])

    def clear(self):
        for metric_name in self.batch_metric_dict:
            self.batch_metric_dict[metric_name] = list()

    def get_batch_averages(self):
        batch_averges = dict()
        for metric_name in self.batch_metric_dict:
            batch_averges[metric_name] = sum(self.batch_metric_dict[metric_name]) / len(self.batch_metric_dict[metric_name])
        return batch_averges
    
    def all_update(self):
        batch_averages = self.get_batch_averages()
        for metric_name in batch_averages:
            self.metric_dict[metric_name].append(batch_averages[metric_name])
        return self.metric_dict
    
    def save(self):
        df = pd.DataFrame.from_dict(self.metric_dict)
        df.to_csv(os.path.join(self.save_dir, "progress.csv"), index=False)

    def all_update_save_clear(self):
        self.all_update()
        self.save()
        self.clear()

