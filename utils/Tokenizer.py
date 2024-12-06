import spacy


class Tokenizer:

    def __init__(self):
        self.spacy_de = spacy.load('de_core_news_sm')  # German
        self.spacy_en = spacy.load('en_core_web_sm')  # English
        self.spacy_fr = spacy.load('fr_core_news_sm')  # French

    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings.
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings.
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def tokenize_fr(self, text):
        """
        Tokenizes French text from a string into a list of strings.
        """
        return [tok.text for tok in self.spacy_fr.tokenizer(text)]
