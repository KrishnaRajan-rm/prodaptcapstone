"""Unit tests for Phase 3: Preprocessing, Tagging, PII Masking, and Chunking."""

import pytest
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from data_preprocessing.preprocessing.tagging import load_aspect_lexicon, build_aspect_regex_patterns
from data_preprocessing.preprocessing.pii_gate1 import load_food_vocab
from data_preprocessing.preprocessing.chunking import chunk_text, split_into_sentences


def test_aspect_lexicon_completeness():
    lexicon = load_aspect_lexicon()
    assert "aspects" in lexicon
    expected_aspects = [
        "wait_time", "food_quality", "staff_behavior", "order_accuracy",
        "cleanliness", "price_value", "portion_size", "ambience_noise",
        "takeout_delivery", "health_safety"
    ]
    for asp in expected_aspects:
        assert asp in lexicon["aspects"]
        assert len(lexicon["aspects"][asp]["patterns"]) >= 30, f"{asp} has fewer than 30 patterns"


def test_food_vocabulary_size():
    vocab = load_food_vocab()
    assert len(vocab) >= 300, f"Food vocab contains {len(vocab)} items, expected >= 300"
    assert "pizza" in vocab
    assert "taco" in vocab
    assert "sushi" in vocab
    assert "cheesesteak" in vocab


def test_sentence_chunking():
    short_text = "The food was amazing. We really enjoyed the evening."
    chunks = chunk_text(short_text)
    assert len(chunks) == 1
    assert chunks[0][0] == short_text

    long_sentences = [f"Sentence number {i} with some descriptive words about dinner." for i in range(50)]
    long_text = " ".join(long_sentences)
    chunks_long = chunk_text(long_text)
    assert len(chunks_long) > 1
    for c_text, n_tok in chunks_long:
        assert n_tok <= 250


def test_vader_sentiment():
    vader = SentimentIntensityAnalyzer()
    neg_score = vader.polarity_scores("The food was cold, awful, and tasted disgusting.")["compound"]
    pos_score = vader.polarity_scores("Outstanding meal, wonderful flavours and excellent host.")["compound"]
    assert neg_score <= -0.05
    assert pos_score >= 0.05
