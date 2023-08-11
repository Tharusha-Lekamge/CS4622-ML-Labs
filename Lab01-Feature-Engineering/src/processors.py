import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize, toktok
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from typing import List
import joblib


class LUTLabelEncoder:
    def __init__(self, labels: List[str]) -> None:
        self.lut = labels

    def transform(self, df: pd.DataFrame) -> np.array:
        enc_lbls = df.apply(lambda st: self.lut.index(st)).to_numpy()
        return enc_lbls

    def inverse_tranform(self, labels: List[int]) -> List[str]:
        labels = [self.lut[lbl] for lbl in labels]
        return labels


class Preprocessor:
    def _strip_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        return text

    def _remove_special_characters(self, text, remove_digits=True):
        pattern = r"[^a-zA-z0-9\s]"
        text = re.sub(pattern, "", text)
        return text

    def _remove_stopwords(self, text, is_lower_case=False):
        tokens = self.tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [
                token for token in tokens if token not in self.stop_words
            ]
        else:
            filtered_tokens = [
                token for token in tokens if token.lower() not in self.stop_words
            ]
        filtered_text = " ".join(filtered_tokens)
        return filtered_text

    def _lemmatize_text(self, text):
        words = word_tokenize(text)
        edited_text = ""
        for word in words:
            lemma_word = self.lemmatizer.lemmatize(word)
            extra = " " + str(lemma_word)
            edited_text += extra
        return edited_text

    def __init__(self):
        # download the corpus
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)

        # initialize
        self.stop_words = set(stopwords.words("english"))
        self.tokenizer = toktok.ToktokTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, txt: str) -> str:
        processed_txt = txt.lower()
        processed_txt = self._strip_html(processed_txt)
        processed_txt = self._remove_special_characters(processed_txt)
        processed_txt = self._remove_stopwords(processed_txt)
        processed_txt = self._lemmatize_text(processed_txt)
        return processed_txt
