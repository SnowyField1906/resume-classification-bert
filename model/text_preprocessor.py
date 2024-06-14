import re
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


class TextPreprocessor:
    def __init__(self):
        self.cached_stop_words = set(stopwords.words("english"))
        self.cached_stop_words.update(
            (
                "and",
                "I",
                "A",
                "http",
                "And",
                "So",
                "arnt",
                "This",
                "When",
                "It",
                "many",
                "Many",
                "so",
                "cant",
                "Yes",
                "yes",
                "No",
                "no",
                "These",
                "these",
                "mailto",
                "regards",
                "ayanna",
                "like",
                "email",
            )
        )

    def remove_stop_words(self, str):
        return " ".join(
            [word for word in str.split() if word not in self.cached_stop_words]
        )

    def punct(self, text):
        token = RegexpTokenizer(r"\w+")
        text = token.tokenize(text)
        return " ".join(text)

    def clean_html(self, text):
        html = re.compile("<.*?>")
        return html.sub(r"", text)

    def remove_links(self, link):
        """Takes a string and removes web links from it"""
        link = re.sub(r"http\S+", "", link)
        link = re.sub(r"bit.ly/\S+", "", link)
        return link.strip("[link]")

    def remove_special_characters(self, text):
        pat = r"[^a-zA-z0-9.,!?/:;\"\'\s]"
        return re.sub(pat, "", text)

    def remove_(self, link):
        link = re.sub("([_]+)", "", link)
        return link

    def remove_digits(self, text):
        pattern = r"[^a-zA-z.,!?/:;\"\'\s]"
        return re.sub(pattern, "", text)

    def lower(self, text):
        return text.lower()

    def email_address(self, text):
        email = re.compile(r"[\w\.-]+@[\w\.-]+")
        return email.sub(r"", text)

    def non_ascii(self, s):
        return "".join(i for i in s if ord(i) < 128)

    def __iter__(self):
        return iter(
            [
                self.remove_stop_words,
                self.punct,
                self.clean_html,
                self.remove_links,
                self.remove_special_characters,
                self.remove_,
                self.remove_digits,
                self.lower,
                self.email_address,
                self.non_ascii,
            ]
        )
