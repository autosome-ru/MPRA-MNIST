import pickle
from Bio import SeqIO
import random
from collections import defaultdict, Counter
import pandas as pd

class MarkovDNA:
    def __init__(self, k=3, alphabet="ACGT", pseudocount=1.0):
        self.k = k
        self.alphabet = list(alphabet)
        self.pseudocount = pseudocount
        self.counts = defaultdict(Counter)
        self.probs = {}

    def fit(self, sequences):
        for seq in sequences:
            seq = seq.upper()
            if len(seq) <= self.k:
                continue
            for i in range(len(seq) - self.k):
                context = seq[i:i+self.k]
                nxt = seq[i+self.k]
                if nxt not in self.alphabet:
                    continue
                self.counts[context][nxt] += 1
        self._normalize()
        return self

    def _normalize(self):
        self.probs = {}
        for context, cnt in self.counts.items():
            total = sum(cnt.values()) + self.pseudocount * len(self.alphabet)
            self.probs[context] = {
                nt: (cnt[nt] + self.pseudocount) / total
                for nt in self.alphabet}

    def parameters(self):
        df = pd.DataFrame.from_dict(
            self.probs,
            orient="index")
        df = df[self.alphabet]
        df.index.name = "context"
        return df.sort_index()

    def generate(self, length, seed=None):
        if seed is None:
            context = random.choice(list(self.probs.keys()))
        else:
            if len(seed) != self.k:
                raise ValueError(f"Seed length must equal k={self.k}")
            context = seed.upper()
        sequence = list(context)
        while len(sequence) < length:
            if context not in self.probs:
                context = random.choice(list(self.probs.keys()))
            probs = self.probs[context]
            nt = random.choices(
                population=self.alphabet,
                weights=[probs[a] for a in self.alphabet],
                k=1
            )[0]
            sequence.append(nt)
            context = "".join(sequence[-self.k:])
        return "".join(sequence)

    def save(self, filename):
        state = {
            "k": self.k,
            "alphabet": self.alphabet,
            "pseudocount": self.pseudocount,
            "probs": self.probs}
        with open(filename, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            state = pickle.load(f)
        model = cls(
            k=state["k"],
            alphabet="".join(state["alphabet"]),
            pseudocount=state["pseudocount"]
        )
        model.probs = state["probs"]
        model.counts = defaultdict(Counter)
        return model