import numpy as np


class FeatureRankings:
    """Tracks feature rankings."""

    def __init__(self, indices, labels=None):

        self.indices = np.array(indices, dtype=int)
        self.labels = np.array(labels, dtype=object)

        # NOTE:
        self.indicators

    def update_votes(self, name, selected):
        """Assign one wote to each selected feature."""

        if self.votes is None:
            self.votes = self.gen_schedule()

        if name not in self.votes.keys():
            self.votes[name] = {}

        for key in selected:
            self.votes[name][str(key)] += 1

        return self

    def update_rankings(self, selected):

        if self.rankings is None:
            self.rankings = self.gen_schedule()

        for key, value in selected.items():
            self.rankings[key] += value

        return self

    def gen_schedule(self):
        """Returns an empty feature votes schedule."""

        entries = [0] * np.size(self.indices)
        return dict(zip(self.indices, entries))


if __name__ == '__main__':

    pass
