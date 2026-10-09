"""Retrieve policy evidence. Keep the instructor's policy_loader.py unchanged."""
import math
import re
from collections import Counter
from policy_loader import load_policy_documents

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = set('a an the this that is are be to for of at on in and or how what which should i my we it do does can with university service desk help please exact'.split())


def tokens(text):
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOP]


class PolicyRetriever:
    def __init__(self):
        # This is the only way documents are loaded, as specified by the exercise.
        self.documents = load_policy_documents()
        self.bags = [Counter(tokens(d.page_content + ' ' + d.metadata.get('topic', '')))
                     for d in self.documents]
        self.df = Counter(word for bag in self.bags for word in bag)

    def search(self, question, k=4):
        query = set(tokens(question))
        n = len(self.documents)
        matches = []
        for doc, bag in zip(self.documents, self.bags):
            score = sum((1.0 + math.log(bag[word])) * math.log(1.0 + (n + 1) / (self.df[word] + 1))
                        for word in query if bag[word])
            if score > 0:
                matches.append((score, doc))
        matches.sort(key=lambda entry: entry[0], reverse=True)
        return [doc for _, doc in matches[:k]]


_retriever = None


def retrieve_policies(question, k=4):
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever.search(question, k)
