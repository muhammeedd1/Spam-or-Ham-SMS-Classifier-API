import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import  WordNetLemmatizer
from nltk.corpus import stopwords

_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS : set[str] | None = None


def _ensure_nltk_resources() -> None :
    resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4"
]
    for resource in resources :
        nltk.download(resource)




def _get_stop_words() -> set[str] :
    global _STOP_WORDS

    if _STOP_WORDS is None :
        _ensure_nltk_resources()
        _STOP_WORDS = set(stopwords.words("english"))

    return _STOP_WORDS


def clean_text(sms: str) -> str:
    sms = re.sub(r"[^a-zA-Z]", " ", sms)
    sms = sms.lower()
    sms = sms.split()
    sms = " ".join(sms)
    return sms


def tokenize_and_lemmatize(cleaned_text) :
    
    _ensure_nltk_resources()

    stop_words = _get_stop_words()

    tokens = nltk.word_tokenize(cleaned_text)
    filterd_words = [word for word in tokens if word not in stop_words]
    lemmas = [_LEMMATIZER.lemmatize(word , pos="v") for word in filterd_words]
    return lemmas


def preprocess(text) :
    cleaned = clean_text(text)
    lemmas = tokenize_and_lemmatize(cleaned)
    return " ".join(lemmas)

