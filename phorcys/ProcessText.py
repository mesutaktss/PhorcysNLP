import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
from typing import List, Optional, Dict, Set, Union
from pathlib import Path
import re
import warnings
from collections import Counter
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from turkish.deasciifier import Deasciifier
DIRPATH = os.path.dirname(os.path.realpath(__file__))
STOPWORDSPATH = DIRPATH + "/data/stopWords.txt"
BADWORDSPATH = DIRPATH + "/data/badWords.txt"
class Normalize:
    STOPWORDS = None
    BADWORDS = None
    def numberToWord(number: int) -> str:
            def convertGroup(number: int, ones: list, tens: list) -> str:
                word = ""
                if number >= 100:
                    if number // 100 != 1:
                        word += ones[number // 100] + " yüz"
                    else:
                        word += "yüz"
                    number = number % 100
                if number >= 10:
                    word += " " + tens[number // 10 - 1]
                    number = number % 10
                if number > 0:
                    word += " " + ones[number]
                return word.strip()
            negativeExpression = None
            if int(number) < 0:
                number = str(number).split("-")[1]
                negativeExpression = "eksi"
                number = int(number)
            ones = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
            tens = ["on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
            scales = ["", "bin", "milyon", "milyar", "trilyon", "katrilyon", "kentilyon", "Sekstilyon", "Septilyon", "Oktilyon", "Nonilyon", "Desilyon", "Undesilyon", "Dodesilyon", "Tredesilyon", "Katordesilyon", "Kendesilyon", "Seksdesilyon", "Septendesilyon", "Oktodesilyon", "Novemdesilyon", "Vigintilyon"]
            word = ""
            if number == 0:
                return ones[0]
            group = 0
            while number > 0:
                number, remainder = divmod(number, 1000)
                if remainder > 0:
                    groupDescription = Normalize.convertGroup(remainder, ones, tens)
                    if group > 0:
                        groupDescription += " " + scales[group]
                    ne = " " if negativeExpression is None else f"{negativeExpression}"
                    word = " " + groupDescription + word
                group += 1
            return ne+word

    @classmethod
    def removeStopwords(cls, text: str, stopwords: Union[Set[str], List[str]] = None) -> str:
        if stopwords is None:
            cls.loadStopwords(STOPWORDSPATH)
            stopwords = cls.STOPWORDS
        elif isinstance(stopwords, list):
            stopwords = set(stopwords)
        cleanedText = " ".join(word for word in text.split() if word.lower() not in stopwords)
        return cleanedText

    @classmethod
    def removeBadWords(cls, text: str, badwords: Union[Set[str], List[str]] = None) -> str:
        if stopwords is None:
            cls.loadStopwords(BADWORDSPATH)
            badwords = cls.BADWORDS
        elif isinstance(badwords, list):
            badwords = set(badwords)
        sortedBadwords = sorted(badwords, key=len, reverse=True)
        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in sortedBadwords) + r')\b', re.IGNORECASE)
        matches = pattern.findall(text)
        numRemoved = len(matches)
        cleanedText = pattern.sub('', text)
        cleanedText = re.sub(' +', ' ', cleanedText).strip()
        return cleanedText, numRemoved

    @classmethod
    def loadStopwords(cls, stop_words_source: Union[str, Set[str], List[str]]) -> None:
        if isinstance(stop_words_source, str):
            with open(stop_words_source, "r", encoding="utf-8") as f:
                cls.STOPWORDS = set(f.read().split())
        elif isinstance(stop_words_source, (set, list)):
            cls.STOPWORDS = set(stop_words_source)
        else:
            raise ValueError(
                "stop_words_source must be a path to a file (str), a set of words (set), or a list of words (list)."
            )

class ProcessText:
    @staticmethod
    def lowerText(text: str) -> str:
        replacements = str.maketrans("IİĞÜŞÖÇ", "ıiğüşöç")
        return text.translate(replacements).lower()

    @staticmethod
    def upperText(text: str) -> str:
        replacements = str.maketrans("ıiğüşöç", "IİĞÜŞÖÇ")
        return text.translate(replacements).upper()

    @staticmethod
    def removePunc(text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod    
    def accentMarkRemove(text: str, accentMap: Optional[Dict[str, str]] = None) -> str:
        if accentMap is None:
            #Default accent map
            accentMap = {"â": "a", "à": "a", "á": "a", "ã": "a", "ä": "a", "å": "a", "ć": "c", "č": "c", "é": "e", "è": "e", "ê": "e", "ë": "e", "î": "i", "ï": "i", "í": "i", "ì": "i", "ñ": "n", "ń": "n", "ô": "o", "ò": "o", "ó": "o", "õ": "o", "ś": "s", "š": "s", "û": "u", "ù": "u", "ú": "u", "ý": "y", "ÿ": "y", "ž": "z", "Â": "A", "À": "A", "Á": "A", "Ã": "A", "Ä": "A", "Å": "A", "Ć": "C", "Č": "C", "É": "E", "È": "E", "Ê": "E", "Ë": "E", "Î": "İ", "Ï": "İ", "Í": "İ", "Ì": "İ", "Ñ": "N", "Ń": "N", "Ô": "O", "Ò": "O", "Ó": "O", "Õ": "O", "Ś": "S", "Š": "S", "Û": "U", "Ù": "U", "Ú": "U", "Ý": "Y", "Ÿ": "Y", "Ž": "Z"}
        translationTable = str.maketrans(accentMap)
        return text.translate(translationTable)

    @staticmethod
    def numToTRText(text):
        def convert_number(match):
            number = float(match.group(0).replace(",", "."))
            if number >= 1e60:
                return warnings.warn(
                    "The number is too big."
                )
            elif number == int(number):
                return Normalize.numberToWord(number=int(number))
            else:
                return warnings.warn(
                    "Decimal numbers are expressed with commas."
                )
        return re.sub(r"[-+]?\d*.\d+|\d+", convert_number, text.replace(",", " virgül ")).lstrip()

    @staticmethod
    def removeNumber(text,signed=True,decimal=True):
        if signed and decimal:
            pattern = r"(?<!\d)[-+]?\d*\.?\d+(?!\d)"
        elif signed:
            pattern = r"(?<!\d)[-+]?\d+(?!\d)"
        elif decimal:
            pattern = r"\d*\.?\d+"
        else:
            pattern = r"\d+"
        text = re.sub(pattern, "", text)
        text = re.sub(r"\s*,\s*", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"^,", "", text).strip()
        return text

    @staticmethod
    def normalizeChars(text, charTable=None):
        if charTable is None:
            charTable = str.maketrans("ğĞıİöÖüÜşŞçÇ", "gGiIoOuUsScC")
        text = text.translate(charTable)
        return text

    @staticmethod
    def wordCounter(text):
        text = num_to_tr_text(ProcessText.removePunc(text))
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return len(words)

    @staticmethod
    def wordExtractor(text):
        text = num_to_tr_text(ProcessText.removePunc(ProcessText.lowerText(text)))
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return words

    @staticmethod
    def sentenceCounter(text):
        nltk.download('punkt')
        sentences = sent_tokenize(text)
        return len(sentences)

    @staticmethod
    def avarageWordCountPerSentence(text):
        totalWords = ProcessText.wordCounter(text)
        totalSentences = ProcessText.sentenceCounter(text)
        return totalWords / totalSentences

    @staticmethod
    def syllableCounter(text):
        text = num_to_tr_text(ProcessText.removePunc(text))
        letters = "aeıioöuüAEIİOÖUÜ"
        return sum(1 for letter in text if letter in letters)

    @staticmethod
    def reabilityTime(text, avaregeReadTime=190):
        totalWords = ProcessText.wordCounter(text)
        readTime = int((totalWords / avaregeReadTime) * 60)
        second = 0
        minute = 0
        hour = 0
        if readTime >= 60:
            minute = readTime // 60 
            second = readTime - (minute * 60)
        else:
            second = readTime
        if minute >= 60:
            hour = minute // 60
            minute = minute - (hour * 60)
        return f"Hour-Saat {hour}, Minute-Dakika {minute}, Second-Saniye {second}"

    @staticmethod
    def readabilityScore(text):
        totalWords = ProcessText.wordCounter(text)
        totalSentences = ProcessText.sentenceCounter(text)
        totalSyllables = ProcessText.syllableCounter(text)
        score = 198.825 - (40.175 * (totalSyllables / totalWords) - 2.610 * (totalSentences / totalWords)) 
        if 80 <= score:
            value = (f"Easy-Kolay\nscore-skor: {score}")
        elif 60 <= score < 80:
            value = (f"Normal\nscore-skor: {score}")
        elif 40 <= score < 60:
            value = (f"Hard-Zor\nscore-skor: {score}")
        elif score < 40:
            value = (f"Complex-Karmaşık\nscore-skor: {score}")
        result = [score, value]
        return result

    @staticmethod
    def frequencyCalculator(text):
        words = ProcessText.wordExtractor(text)
        frequency = Counter(words)
        return frequency

    @staticmethod
    def phoneticTransform(text):
        phoneticDict = {"a": "a", "b": "b", "c": "d͡ʒ", "ç": "t͡ʃ", "d": "d", "e": "e", "f": "f", "g": "g", "ğ": "ɰ", "h": "h", "ı": "ɯ", "i": "i", "j": "ʒ", "k": "k", "l": "l", "m": "m", "n": "n", "o": "o", "ö": "œ", "p": "p", "r": "r", "s": "s", "ş": "ʃ", "t": "t", "u": "u", "ü": "y", "v": "v", "y": "j", "z": "z", "q": "q", "w": "w", "x": "ks"}
        translationTable = str.maketrans(phoneticDict)
        return text.translate(translationTable)

    @staticmethod
    def sentenceTokenizer(text):
        nltk.download('punkt')
        sentences = sent_tokenize(text)
        return sentences

    @staticmethod
    def findIdioms(text):
        text = ProcessText.lowerText(text)
        def loadIdioms():
            idiomsFile = 'data/idioms.txt'
            with open(idiomsFile, 'r', encoding='utf-8') as file:
                idiomsR = file.read().splitlines()
            return idiomsR
        datas = loadIdioms()
        idioms = [idiom for idiom in datas if idiom in text]
        return idioms

    @staticmethod
    def calculateSimilarity(text1, text2, modelName="dbmdz/bert-base-turkish-cased"):
        from transformers import AutoTokenizer, AutoModel
        import torch
        from sklearn.metrics.pairwise import cosine_similarity
        model_name = modelName
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        def get_sentence_embedding(sentence):
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            sentence_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            return sentence_embedding
        embedding1 = get_sentence_embedding(text1)
        embedding2 = get_sentence_embedding(text2)
        similarity = cosine_similarity(embedding1, embedding2)
        return similarity[0][0]

    @staticmethod
    def deasciify(text):
        return Deasciifier(text).convert_to_turkish()
