"""Streamlit Virtual Lab: Text Preprocessing, Normalization & Boolean Retrieval.

A 7-section virtual laboratory application:
1. Purpose: Aim, learning objectives, scope, prerequisites, and real-world significance.
2. Theory: Preprocessing fundamentals, tokenization, stop-words, morphology, and Boolean retrieval theory.
3. Simulation: Interactive pipeline execution on chunked tokens, token frequency charts, Boolean retrieval evaluation, and trial logging.
4. Quiz: Self-grading conceptual assessment with randomized questions and explanations.
5. Report Generation: Comprehensive lab report with student metadata, comparative analysis, and downloadable PDF with embedded graphs.
6. Certificate: Dynamic, verifiable Certificate of Completion with downloadable PDF.
7. References: Academic citations, textbook readings, and virtual lab documentation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import random
import re
import unicodedata

from fpdf import FPDF
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ======================================================================================
# 1. CUSTOM STYLES & THEMING (SIMPLE, THEME-AWARE & READABLE)
# ======================================================================================

def inject_custom_styles() -> None:
    """Inject clean, simple CSS that increases readability and adapts to both light and dark mode."""
    st.markdown(
        """
        <style>
        /* Typography: Increase font size cleanly across all themes */
        html, body, [class*="css"], .stMarkdown, p, div, span, label, li {
            font-size: 1.05rem;
            line-height: 1.6;
        }

        /* Headings - respect current dark/light theme */
        h1, .stMarkdown h1 {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }
        h2, .stMarkdown h2 {
            font-size: 1.65rem !important;
            font-weight: 600 !important;
            padding-bottom: 6px;
            margin-top: 14px;
            margin-bottom: 12px;
        }
        h3, .stMarkdown h3 {
            font-size: 1.32rem !important;
            font-weight: 600 !important;
            margin-top: 12px;
            margin-bottom: 8px;
        }
        h4, .stMarkdown h4 {
            font-size: 1.15rem !important;
            font-weight: 600 !important;
        }

        /* Form and widget labels */
        .stRadio label, .stCheckbox label, .stSelectbox label, 
        .stTextInput label, .stTextArea label, .stFileUploader label {
            font-size: 1.02rem !important;
            font-weight: 500;
        }

        /* Buttons: comfortable sizing */
        .stButton > button, .stDownloadButton > button {
            font-size: 1rem !important;
            font-weight: 500 !important;
            border-radius: 6px !important;
        }

        /* Tables and DataFrames: center text & enable horizontal overflow scrolling */
        .stDataFrame, .stTable {
            font-size: 0.98rem !important;
            overflow-x: auto !important;
        }

        .stTable {
            overflow-x: auto !important;
            display: block !important;
            width: 100% !important;
        }
        .stTable table {
            width: 100% !important;
            border-collapse: collapse !important;
        }
        .stTable th, .stTable td {
            text-align: center !important;
            vertical-align: middle !important;
            white-space: nowrap !important;
        }

        div[data-testid="stDataFrame"] {
            overflow-x: auto !important;
            max-width: 100% !important;
        }
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] div[class*="cell"],
        div[data-testid="stDataFrame"] td,
        div[data-testid="stDataFrame"] th,
        [data-testid="stSidebar"] [data-testid="stDataFrame"] div[role="gridcell"],
        [data-testid="stSidebar"] [data-testid="stDataFrame"] div[role="columnheader"] {
            text-align: center !important;
            justify-content: center !important;
            align-items: center !important;
        }

        /* Custom Emerald Green Record Trial Button */
        .record-trial-btn > button,
        .record-trial-btn button {
            background-color: #10b981 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            box-shadow: 0 2px 5px rgba(16, 185, 129, 0.25) !important;
            transition: all 0.2s ease-in-out !important;
        }
        .record-trial-btn > button:hover,
        .record-trial-btn button:hover {
            background-color: #059669 !important;
            color: #ffffff !important;
            box-shadow: 0 4px 10px rgba(5, 150, 105, 0.35) !important;
        }

        /* Seamless Hover Tooltip for Key Terms (looks identical to normal text) */
        .term-tooltip {
            position: relative;
            display: inline-block;
            color: inherit;
            background: transparent;
            padding: 0;
            margin: 0;
            border: none;
            font-weight: inherit;
            cursor: pointer;
        }
        .term-tooltip:hover {
            color: #2563eb;
        }

        .term-tooltip .tooltip-box {
            visibility: hidden;
            opacity: 0;
            width: 280px;
            background-color: var(--secondary-background-color, #1e293b);
            color: var(--text-color, #f8fafc);
            text-align: left;
            border: 1px solid rgba(59, 130, 246, 0.4);
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 10px 14px;
            position: absolute;
            z-index: 99999;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
            font-size: 0.88rem;
            font-weight: 400;
            line-height: 1.45;
            white-space: normal;
            transition: opacity 0.2s ease-in-out, visibility 0.2s ease-in-out, transform 0.2s ease-in-out;
            pointer-events: none;
        }

        /* Dialog Box Arrow */
        .term-tooltip .tooltip-box::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -6px;
            border-width: 6px;
            border-style: solid;
            border-color: rgba(59, 130, 246, 0.4) transparent transparent transparent;
        }

        .term-tooltip:hover .tooltip-box {
            visibility: visible;
            opacity: 1;
            transform: translateX(-50%) translateY(-4px);
        }

        /* Metric cards - adapt cleanly to dark and light modes */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color) !important;
            border: 1px solid rgba(128, 128, 128, 0.2) !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.95rem !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.98rem !important;
        }

        /* Clean top border accents on the 4 metric cards */
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4)) > div:nth-child(1) div[data-testid="stMetric"] {
            border-top: 3px solid #0284c7 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4)) > div:nth-child(2) div[data-testid="stMetric"] {
            border-top: 3px solid #6366f1 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4)) > div:nth-child(3) div[data-testid="stMetric"] {
            border-top: 3px solid #a855f7 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(> div:nth-child(4)) > div:nth-child(4) div[data-testid="stMetric"] {
            border-top: 3px solid #10b981 !important;
        }

        /* Analogy Card - simple and theme-adaptive */
        .analogy-card {
            background-color: var(--secondary-background-color);
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 18px 20px;
            margin: 14px 0;
            border-top: 1px solid rgba(128, 128, 128, 0.15);
            border-right: 1px solid rgba(128, 128, 128, 0.15);
            border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        }
        .analogy-title {
            font-weight: 600;
            color: #3b82f6;
            margin-bottom: 8px;
            font-size: 1.05rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================================
# 2. CONSTANTS, STOP WORDS & MORPHOLOGY DICTIONARIES
# ======================================================================================

EXPERIMENT = {
    "number": 2,
    "title": "Text Preprocessing, Normalization & Boolean Retrieval",
    "estimated_minutes": "6–10 minutes",
    "aim": (
        "To perform tokenization, noise removal, stop-word filtering, stemming, "
        "and lemmatization on chunked token streams, and evaluate Boolean retrieval on the preprocessed corpus."
    ),
    "expected_outcome": (
        "A standardized, noise-free token vocabulary and exact Boolean document retrieval (AND, OR, NOT)."
    ),
    "objectives": [
        "Preprocess raw, unnormalized token sequences (comma-separated or text streams).",
        "Apply case folding, punctuation removal, contraction expansion, and stop-word filtering.",
        "Compare rule-based stemming against morphological lemmatization.",
        "Evaluate exact Boolean retrieval queries (AND, OR, NOT) across preprocessed document chunks.",
        "Analyze vocabulary compression metrics and generate an official laboratory report with graphs.",
    ],
    "prerequisites": [
        "Basic understanding of Python strings and lists.",
        "Elementary set theory (intersection, union, set difference).",
        "Foundational concepts of Search Engines and Information Retrieval.",
    ],
}

DEFAULT_TOKENS = (
    "information,retrieval,systems,process,text,and,documents,efficiently,"
    "search,engines,index,text,documents,for,fast,information,retrieval,"
    "text,preprocessing,helps,information,retrieval,by,filtering,stopwords,"
    "boolean,retrieval,models,use,and,or,not,operators,on,indexed,documents,"
    "retrieval,systems,don't,keep,noisy,text,in,the,search,index,"
    "researchers,are,studying,advanced,information,retrieval,and,text,mining,"
    "lemmatization,and,stemming,normalize,retrieved,text,tokens,for,retrieval,"
    "a,smart,retrieval,system,searches,documents,and,returns,relevant,data"
)

KEY_TERMS = {
    "Tokenization": "The process of breaking continuous text into individual words or atomic units called tokens.",
    "Case Folding": "Converting all letters to lowercase so uppercase and lowercase words (e.g., Text vs text) match identically.",
    "Contraction Expansion": "Expanding short forms like 'don't' to 'do not' before punctuation is stripped to preserve underlying meaning.",
    "Noise Removal": "Stripping special symbols, diacritical accents, numbers, and punctuation marks to leave clean text.",
    "Stop-Word Filtering": "Removing high-frequency filler words (the, is, at, and) that add noise without contributing unique search meaning.",
    "Stemming": "Fast, rule-based suffix stripping that reduces words to stems (e.g. studying ➔ studi). Stems may not be valid words.",
    "Lemmatization": "Dictionary-based morphological reduction that maps inflected words to valid dictionary base forms (e.g. studying ➔ study, were ➔ be).",
    "Boolean Retrieval": "An exact-match search model based on set theory that evaluates queries with AND, OR, and NOT operators over preprocessed tokens.",
    "AND Operator": "Intersection set operation (∩): matches documents containing BOTH search terms (e.g. information AND retrieval).",
    "OR Operator": "Union set operation (∪): matches documents containing AT LEAST ONE of the search terms (e.g. text OR search).",
    "NOT Operator": "Set difference operation (\\): excludes documents containing an unwanted term (e.g. retrieval AND NOT noisy).",
    "Vocabulary Size": "The total count of distinct, unique preprocessed tokens across a document corpus.",
    "Inverted Index": "A search engine data structure that maps each unique preprocessed token to a list of document IDs containing it.",
}


def key_term_html(term: str, custom_label: str | None = None) -> str:
    """Return an interactive HTML span with a hover dialog box popup for key terms."""
    explanation = KEY_TERMS.get(term, "Information Retrieval concept.")
    label = custom_label if custom_label else term
    return (
        f'<span class="term-tooltip">{label}'
        f'<span class="tooltip-box"><b>{term}:</b> {explanation}</span>'
        f'</span>'
    )

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she", "that",
    "the", "their", "theirs", "them", "they", "this", "those", "to", "was", "we",
    "were", "will", "with", "you", "your", "yours",
}

CONTRACTIONS = {
    "can't": "cannot", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "isn't": "is not",
    "shouldn't": "should not", "wasn't": "was not", "weren't": "were not",
    "won't": "will not", "wouldn't": "would not", "i'm": "i am",
    "it's": "it is", "they're": "they are", "we're": "we are", "you're": "you are",
}

IRREGULAR_LEMMAS = {
    "am": "be", "are": "be", "is": "be", "was": "be", "were": "be",
    "been": "be", "children": "child", "feet": "foot", "geese": "goose",
    "men": "man", "mice": "mouse", "teeth": "tooth", "women": "woman",
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "went": "go", "gone": "go", "ran": "run", "written": "write",
    "studying": "study", "studies": "study", "studied": "study",
    "entities": "entity", "systems": "system", "documents": "document",
    "researchers": "researcher", "words": "word", "graphs": "graph",
}


# ======================================================================================
# 2. PREPROCESSING PIPELINE DATA STRUCTURES & FUNCTIONS
# ======================================================================================

@dataclass(frozen=True)
class PipelineOptions:
    lowercase: bool = True
    expand_contractions: bool = True
    remove_accents: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = True
    morphology: str = "Lemmatization"


@dataclass(frozen=True)
class PipelineResult:
    original: str
    normalized: str
    tokens: list[str]
    filtered_tokens: list[str]
    final_tokens: list[str]
    output_text: str
    removed_stopwords: list[str]
    original_token_count: int
    final_token_count: int
    vocabulary_size: int
    reduction_percent: float
    frequencies: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_text(text: str, options: PipelineOptions) -> str:
    """Apply configurable surface-level normalization."""
    value = text.replace(",", " ")
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u2018", "'").replace("\u2019", "'")

    if options.lowercase:
        value = value.lower()

    if options.expand_contractions:
        for contraction, expanded in sorted(
            CONTRACTIONS.items(), key=lambda item: len(item[0]), reverse=True
        ):
            value = re.sub(rf"\b{re.escape(contraction)}\b", expanded, value, flags=re.I)

    if options.remove_accents:
        value = "".join(
            char
            for char in unicodedata.normalize("NFKD", value)
            if not unicodedata.combining(char)
        )

    if options.remove_numbers:
        value = re.sub(r"\b\d+(?:[.,]\d+)?\b", " ", value)

    if options.remove_punctuation:
        value = re.sub(r"[^\w\s'-]", " ", value)
        value = value.replace("_", " ").replace("'", " ").replace("-", " ")

    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str) -> list[str]:
    """Split text/tokens on commas or whitespace while preserving valid alphanumeric tokens."""
    cleaned = text.replace(",", " ")
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?", cleaned)


def stem_word(word: str) -> str:
    """Apply rule-based suffix stemming."""
    val = word.lower()
    if len(val) <= 3:
        return val
    if val.endswith("ies") and len(val) > 4:
        return val[:-3] + "y"
    if val.endswith("ing") and len(val) > 5:
        base = val[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if val.endswith("ed") and len(val) > 4:
        base = val[:-2]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    for suffix in ("ization", "ational", "fulness", "ousness", "iveness", "ment", "ness", "ly"):
        if val.endswith(suffix) and len(val) - len(suffix) >= 3:
            return val[: -len(suffix)] + ("ize" if suffix == "ization" else "")
    if val.endswith("es") and len(val) > 4:
        return val[:-2]
    if val.endswith("s") and not val.endswith("ss") and len(val) > 3:
        return val[:-1]
    return val


def lemmatize_word(word: str) -> str:
    """Return dictionary/rule-based morphological base form."""
    val = word.lower()
    if val in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[val]
    if val.endswith("ies") and len(val) > 4:
        return val[:-3] + "y"
    if val.endswith("ves") and len(val) > 4:
        return val[:-3] + "f"
    if val.endswith("ing") and len(val) > 5:
        base = val[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        if base.endswith(("at", "iz", "bl", "ud", "fi")):
            base += "e"
        return base
    if val.endswith("ed") and len(val) > 4:
        base = val[:-2]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if val.endswith("es") and len(val) > 4 and val[-3] in "sxz":
        return val[:-2]
    if val.endswith("s") and not val.endswith(("ss", "us", "is")) and len(val) > 3:
        return val[:-1]
    return val


def run_pipeline(text: str, options: PipelineOptions) -> PipelineResult:
    """Execute the full text preprocessing pipeline and expose intermediate stages."""
    normalized = normalize_text(text, options)
    tokens = tokenize(normalized)
    removed = [t for t in tokens if options.remove_stopwords and t.lower() in STOP_WORDS]
    filtered = [
        t for t in tokens
        if not options.remove_stopwords or t.lower() not in STOP_WORDS
    ]

    if options.morphology == "Stemming":
        final_tokens = [stem_word(t) for t in filtered]
    elif options.morphology == "Lemmatization":
        final_tokens = [lemmatize_word(t) for t in filtered]
    else:
        final_tokens = filtered.copy()

    original_count = len(tokenize(text))
    final_count = len(final_tokens)
    reduction = ((original_count - final_count) / original_count * 100) if original_count else 0.0

    return PipelineResult(
        original=text,
        normalized=normalized,
        tokens=tokens,
        filtered_tokens=filtered,
        final_tokens=final_tokens,
        output_text=" ".join(final_tokens),
        removed_stopwords=removed,
        original_token_count=original_count,
        final_token_count=final_count,
        vocabulary_size=len(set(final_tokens)),
        reduction_percent=round(reduction, 2),
        frequencies=dict(Counter(final_tokens).most_common()),
    )


def split_into_chunks(text: str, chunk_size: int = 8) -> list[str]:
    """Split raw text or token lists into chunked documents for retrieval evaluation."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    tokens = tokenize(text)
    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        return [", ".join(tokens)]

    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = ", ".join(tokens[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# ======================================================================================
# 3. BOOLEAN RETRIEVAL ENGINE
# ======================================================================================

def tokenize_boolean_query(query: str) -> list[str]:
    """Tokenize a boolean expression into operators (AND, OR, NOT), parentheses, and terms."""
    cleaned = query.replace("(", " ( ").replace(")", " ) ")
    raw_tokens = cleaned.split()
    tokens = []
    for tok in raw_tokens:
        upper = tok.upper()
        if upper in {"AND", "OR", "NOT", "(", ")"}:
            tokens.append(upper)
        else:
            tokens.append(tok)
    return tokens


def evaluate_boolean_query(
    query: str,
    documents: list[str],
    options: PipelineOptions,
) -> tuple[set[int], list[str]]:
    """
    Evaluate a Boolean Query with AND, OR, NOT and parentheses over preprocessed documents.
    Returns (set of matching 1-indexed doc IDs, list of step-by-step logic traces).
    """
    all_doc_ids = set(range(1, len(documents) + 1))
    traces: list[str] = []

    doc_tokens: dict[int, set[str]] = {}
    for doc_id, doc in enumerate(documents, start=1):
        p_res = run_pipeline(doc, options)
        doc_tokens[doc_id] = set(p_res.final_tokens)

    def term_postings(term: str) -> set[int]:
        p_term = run_pipeline(term, options).final_tokens
        if not p_term:
            return set()
        target = p_term[0]
        matches = {doc_id for doc_id, tokens in doc_tokens.items() if target in tokens}
        traces.append(f"Term '{term}' (processed: '{target}') -> Found in: {sorted(list(matches)) or 'None'}")
        return matches

    tokens = tokenize_boolean_query(query)
    if not tokens:
        return set(), ["Empty query."]

    precedence = {"NOT": 3, "AND": 2, "OR": 1}
    postfix: list[str] = []
    op_stack: list[str] = []

    normalized_tokens: list[str] = []
    for i, tok in enumerate(tokens):
        normalized_tokens.append(tok)
        if i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            is_curr_term = tok not in {"AND", "OR", "NOT", "("}
            is_next_term = next_tok not in {"AND", "OR", ")"}
            if is_curr_term and is_next_term:
                normalized_tokens.append("AND")

    for tok in normalized_tokens:
        if tok in precedence:
            while (
                op_stack
                and op_stack[-1] != "("
                and precedence.get(op_stack[-1], 0) >= precedence[tok]
            ):
                postfix.append(op_stack.pop())
            op_stack.append(tok)
        elif tok == "(":
            op_stack.append(tok)
        elif tok == ")":
            while op_stack and op_stack[-1] != "(":
                postfix.append(op_stack.pop())
            if op_stack and op_stack[-1] == "(":
                op_stack.pop()
        else:
            postfix.append(tok)

    while op_stack:
        postfix.append(op_stack.pop())

    eval_stack: list[set[int]] = []
    for tok in postfix:
        if tok == "NOT":
            if eval_stack:
                operand = eval_stack.pop()
                result = all_doc_ids - operand
                traces.append(f"Operation NOT -> {sorted(list(result))}")
                eval_stack.append(result)
            else:
                eval_stack.append(all_doc_ids)
        elif tok == "AND":
            if len(eval_stack) >= 2:
                right = eval_stack.pop()
                left = eval_stack.pop()
                result = left & right
                traces.append(f"Operation AND ({sorted(list(left))} ∩ {sorted(list(right))}) -> {sorted(list(result))}")
                eval_stack.append(result)
        elif tok == "OR":
            if len(eval_stack) >= 2:
                right = eval_stack.pop()
                left = eval_stack.pop()
                result = left | right
                traces.append(f"Operation OR ({sorted(list(left))} ∪ {sorted(list(right))}) -> {sorted(list(result))}")
                eval_stack.append(result)
        else:
            eval_stack.append(term_postings(tok))

    final_matches = eval_stack[0] if eval_stack else set()
    return final_matches, traces


# ======================================================================================
# 4. QUIZ QUESTION BANK (PREPROCESSING & BOOLEAN RETRIEVAL)
# ======================================================================================

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "Which operation breaks continuous raw text into discrete words or units?",
        "options": ["Tokenization", "Stemming", "Boolean AND", "Normalization"],
        "answer": 0,
        "explanation": "Tokenization segments text into individual tokens for downstream preprocessing.",
    },
    {
        "id": 2,
        "question": "Why are stop words frequently filtered during preprocessing?",
        "options": [
            "To encrypt document contents",
            "To remove high-frequency function words that add little discriminative meaning",
            "To correct spelling mistakes",
            "To convert text into binary numbers",
        ],
        "answer": 1,
        "explanation": "Stop words (e.g., 'the', 'is', 'at') occur with high frequency and can clutter vocabularies.",
    },
    {
        "id": 3,
        "question": "What is the primary difference between Stemming and Lemmatization?",
        "options": [
            "Stemming removes stop words, lemmatization converts case",
            "Stemming uses affix rules (stems may not be words); lemmatization uses vocabulary & grammar to return valid lemmas",
            "Stemming is slower and requires a dictionary; lemmatization is rule-based",
            "They produce identical results in all cases",
        ],
        "answer": 1,
        "explanation": "Stemming slices off suffixes mechanically, whereas lemmatization resolves tokens to meaningful dictionary lemmas.",
    },
    {
        "id": 4,
        "question": "In a Boolean Retrieval system, what does the query 'information AND retrieval' return?",
        "options": [
            "Documents containing either 'information' or 'retrieval'",
            "Documents containing both 'information' and 'retrieval'",
            "Documents containing 'information' but not 'retrieval'",
            "All documents in the corpus",
        ],
        "answer": 1,
        "explanation": "The Boolean AND operator computes the intersection (∩) of documents containing both terms.",
    },
    {
        "id": 5,
        "question": "In Boolean Retrieval, which set operation corresponds to the OR operator?",
        "options": ["Intersection", "Set Difference", "Union", "Cartesian Product"],
        "answer": 2,
        "explanation": "The OR operator takes the union (∪) of documents containing at least one of the queried terms.",
    },
    {
        "id": 6,
        "question": "Why should contractions (e.g., 'don't') be expanded before punctuation is stripped?",
        "options": [
            "Removing punctuation first would turn 'don't' into 'dont', obscuring 'do not'",
            "It triples the vocabulary size",
            "It is required by the Boolean model",
            "Order of preprocessing operations has no effect",
        ],
        "answer": 0,
        "explanation": "Expanding contractions preserves underlying lexical components ('do not') before apostrophes are lost.",
    },
    {
        "id": 7,
        "question": "What does the query 'student AND NOT noisy' return?",
        "options": [
            "All documents with 'noisy'",
            "Documents containing 'student' that do NOT contain 'noisy'",
            "Documents containing neither term",
            "Documents containing both terms",
        ],
        "answer": 1,
        "explanation": "AND NOT computes set difference: documents with 'student' minus documents containing 'noisy'.",
    },
    {
        "id": 8,
        "question": "What is Case Folding in text preprocessing?",
        "options": [
            "Converting text to a uniform case (usually lowercase) so case variations match",
            "Dividing text into equal halves",
            "Compressing text into zip format",
            "Translating words into another language",
        ],
        "answer": 0,
        "explanation": "Case folding maps capitalized words (e.g., 'Text', 'TEXT') to lowercase ('text') for consistent matching.",
    },
    {
        "id": 9,
        "question": "What is the result of lemmatizing the irregular verb 'were'?",
        "options": ["wer", "were", "be", "being"],
        "answer": 2,
        "explanation": "The lemma of inflected verb forms 'am', 'is', 'are', 'was', 'were' is 'be'.",
    },
    {
        "id": 10,
        "question": "How does preprocessing impact Boolean retrieval performance?",
        "options": [
            "It ensures search terms and document tokens share identical normalized forms (e.g. 'systems' matches 'system')",
            "It turns every document into an image",
            "It permanently prevents any search queries from executing",
            "It increases vocabulary size to infinity",
        ],
        "answer": 0,
        "explanation": "Applying identical preprocessing to queries and documents bridges morphological variations.",
    },
    {
        "id": 11,
        "question": "Which of the following is an example of an English stop word?",
        "options": ["retrieval", "the", "algorithm", "database"],
        "answer": 1,
        "explanation": "'the' is a common functional stop word.",
    },
    {
        "id": 12,
        "question": "What is vocabulary size after preprocessing?",
        "options": [
            "Total number of characters in the corpus",
            "The count of unique, distinct preprocessed tokens in the corpus",
            "The total number of sentences",
            "The number of times the letter 'e' appears",
        ],
        "answer": 1,
        "explanation": "Vocabulary size represents the cardinality of the unique token set.",
    },
    {
        "id": 13,
        "question": "If Doc1 has {study, text} and Doc2 has {system, text}, which documents match 'study OR system'?",
        "options": ["Only Doc1", "Only Doc2", "Both Doc1 and Doc2", "Neither document"],
        "answer": 2,
        "explanation": "Doc1 contains 'study' and Doc2 contains 'system', so their union contains both Doc1 and Doc2.",
    },
    {
        "id": 14,
        "question": "What is over-stemming?",
        "options": [
            "When two unrelated words are reduced to the same root (false positive match)",
            "When words are not stemmed enough",
            "When stop words are not removed",
            "When case folding fails",
        ],
        "answer": 0,
        "explanation": "Over-stemming occurs when distinct words (e.g., 'organization' and 'organ') are conflated to the same stem.",
    },
    {
        "id": 15,
        "question": "In a Boolean expression '(A OR B) AND C', which operation is evaluated first?",
        "options": [
            "The expression inside parentheses (A OR B)",
            "AND C",
            "Random order",
            "NOT operation",
        ],
        "answer": 0,
        "explanation": "Parentheses have the highest precedence in Boolean expressions.",
    },
]


# ======================================================================================
# 5. STATE MANAGEMENT & HELPER FUNCTIONS
# ======================================================================================

def initialize_state() -> None:
    defaults = {
        "trials": [],
        "quiz_answers": {},
        "quiz_submitted": False,
        "quiz_score": 0,
        "quiz_question_set": None,
        "student_info": {"name": "", "roll": "", "date": str(date.today())},
        "student_notes": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def configuration_label(options: PipelineOptions) -> str:
    enabled = []
    if options.lowercase:
        enabled.append("lowercase")
    if options.expand_contractions:
        enabled.append("contractions")
    if options.remove_accents:
        enabled.append("accents")
    if options.remove_punctuation:
        enabled.append("punctuation")
    if options.remove_numbers:
        enabled.append("numbers")
    if options.remove_stopwords:
        enabled.append("stopwords")
    enabled.append(options.morphology.lower())
    return ", ".join(enabled)


def read_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    if uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="replace")
    df = pd.read_csv(uploaded_file)
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if text_cols:
        return " ".join(df[text_cols[0]].dropna().astype(str).tolist())
    return ",".join(df.astype(str).values.flatten())


def make_pipeline_flow_figure(morphology: str) -> go.Figure:
    labels = ["Raw Tokens", "Normalization", "Tokenization", "Stop-words", morphology, "Clean Corpus"]
    x_vals = list(range(len(labels)))
    fig = go.Figure()
    for idx in range(len(labels) - 1):
        fig.add_trace(
            go.Scatter(
                x=[x_vals[idx], x_vals[idx + 1]],
                y=[0, 0],
                mode="lines",
                line={"color": "#94a3b8", "width": 3},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=[0] * len(labels),
            mode="markers+text",
            marker={
                "size": 34,
                "color": ["#f59e0b", "#3b82f6", "#2563eb", "#1d4ed8", "#7c3aed", "#16a34a"],
            },
            text=labels,
            textposition="bottom center",
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=170,
        margin={"l": 20, "r": 20, "t": 15, "b": 50},
        xaxis={"visible": False, "range": [-0.5, len(labels) - 0.5]},
        yaxis={"visible": False, "range": [-0.4, 0.3]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ======================================================================================
# 6. PDF LAB REPORT & CERTIFICATE GENERATOR WITH EMBEDDED CHARTS
# ======================================================================================

class LabReportPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report", align="C")


def pdf_safe(value: object) -> str:
    return str(value).encode("latin-1", "replace").decode("latin-1")


def pdf_section_title(pdf: LabReportPDF, number: int, title: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 64, 175)
    pdf.cell(0, 7, f"{number}. {title}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(31, 41, 55)


def draw_pdf_tokens_chart(pdf: LabReportPDF, trials: list[dict]) -> None:
    """Draw a clean grouped bar chart of Input vs Final Tokens in PDF."""
    if not trials:
        return
    chart_trials = trials[:6]
    max_tokens = max(
        max(float(t.get("Input Tokens", 0)), float(t.get("Final Tokens", 0)))
        for t in chart_trials
    ) or 1

    chart_x = 22
    chart_y = pdf.get_y() + 4
    chart_w = 165
    chart_h = 42
    grp_w = chart_w / len(chart_trials)

    pdf.set_draw_color(148, 163, 184)
    pdf.line(chart_x, chart_y, chart_x, chart_y + chart_h)
    pdf.line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h)

    for i, t in enumerate(chart_trials):
        in_val = float(t.get("Input Tokens", 0))
        out_val = float(t.get("Final Tokens", 0))
        in_h = (in_val / max_tokens) * (chart_h - 6)
        out_h = (out_val / max_tokens) * (chart_h - 6)

        grp_x = chart_x + i * grp_w + grp_w * 0.15
        bar_w = max(4, grp_w * 0.28)

        pdf.set_fill_color(148, 163, 184)
        pdf.rect(grp_x, chart_y + chart_h - in_h, bar_w, in_h, "F")

        pdf.set_fill_color(37, 99, 235)
        pdf.rect(grp_x + bar_w + 1.5, chart_y + chart_h - out_h, bar_w, out_h, "F")

        pdf.set_xy(grp_x - 1, chart_y + chart_h + 1)
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(grp_w * 0.8, 4, f"T{t.get('Trial #', i + 1)}", align="C")

    pdf.set_xy(chart_x, chart_y + chart_h + 7)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_fill_color(148, 163, 184)
    pdf.rect(chart_x, pdf.get_y() + 1, 3, 3, "F")
    pdf.cell(5, 5, "")
    pdf.cell(30, 5, "Input tokens")

    pdf.set_fill_color(37, 99, 235)
    pdf.rect(chart_x + 40, pdf.get_y() + 1, 3, 3, "F")
    pdf.cell(5, 5, "")
    pdf.cell(30, 5, "Final tokens", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(chart_y + chart_h + 14)


def draw_pdf_reduction_chart(pdf: LabReportPDF, trials: list[dict]) -> None:
    """Draw a bar chart of Token Reduction Percentage across trials in PDF."""
    if not trials:
        return
    chart_trials = trials[:6]
    max_red = max(float(t.get("Reduction (%)", 0)) for t in chart_trials) or 100
    max_scale = max(100.0, max_red)

    chart_x = 22
    chart_y = pdf.get_y() + 4
    chart_w = 165
    chart_h = 35
    grp_w = chart_w / len(chart_trials)

    pdf.set_draw_color(148, 163, 184)
    pdf.line(chart_x, chart_y, chart_x, chart_y + chart_h)
    pdf.line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h)

    for i, t in enumerate(chart_trials):
        red_val = float(t.get("Reduction (%)", 0))
        bar_h = (red_val / max_scale) * (chart_h - 6)
        grp_x = chart_x + i * grp_w + grp_w * 0.25
        bar_w = max(5, grp_w * 0.5)

        pdf.set_fill_color(22, 163, 74)
        pdf.rect(grp_x, chart_y + chart_h - bar_h, bar_w, bar_h, "F")

        pdf.set_xy(grp_x - 2, chart_y + chart_h - bar_h - 4)
        pdf.set_font("Helvetica", "B", 6)
        pdf.cell(bar_w + 4, 3, f"{red_val:.1f}%", align="C")

        pdf.set_xy(grp_x - 1, chart_y + chart_h + 1)
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(bar_w + 2, 4, f"T{t.get('Trial #', i + 1)}", align="C")

    pdf.set_y(chart_y + chart_h + 8)


def generate_pdf_report(
    student_name: str,
    roll_number: str,
    experiment_date: str,
    trials: list[dict],
    notes: str,
    quiz_score: int = 0,
    total_quiz_questions: int = 10,
    latest_query: str = "",
    latest_matches: str = "",
) -> bytes:
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(30, 64, 175)
    pdf.multi_cell(0, 8, EXPERIMENT["title"], new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(31, 41, 55)

    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, pdf_safe(f"Experiment No.: {EXPERIMENT['number']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, pdf_safe(f"Student Name: {student_name or 'N/A'} | Roll No: {roll_number or 'N/A'}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, pdf_safe(f"Date: {experiment_date} | Quiz Score: {quiz_score}/{total_quiz_questions}"), new_x="LMARGIN", new_y="NEXT")

    # 1. Aim & Objectives
    pdf_section_title(pdf, 1, "Aim & Objectives")
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, pdf_safe(EXPERIMENT["aim"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for obj in EXPERIMENT["objectives"]:
        pdf.multi_cell(0, 4.5, pdf_safe(f"- {obj}"), new_x="LMARGIN", new_y="NEXT")

    # 2. Methodology
    pdf_section_title(pdf, 2, "Preprocessing Pipeline & Boolean Retrieval Methodology")
    pdf.set_font("Helvetica", "", 8.5)
    methodology = [
        "1. Input Stream: Raw chunked tokens (comma-separated or text streams) are ingested without prior preprocessing.",
        "2. Normalization: Case folding to lowercase, contraction expansion ('don't' -> 'do not'), accent stripping, and noise filtering.",
        "3. Tokenization & Stop-word Filtering: Segmentation of tokens and removal of high-frequency low-discriminative terms.",
        "4. Morphological Transformation: Rule-based suffix Stemming or dictionary-based Lemmatization to reduce inflectional forms.",
        "5. Boolean Retrieval: Evaluation of compound Boolean queries (AND, OR, NOT, parentheses) across preprocessed document chunks.",
    ]
    for step in methodology:
        pdf.multi_cell(0, 4.5, pdf_safe(step), new_x="LMARGIN", new_y="NEXT")

    # 3. Experimental Results
    pdf_section_title(pdf, 3, "Experimental Trials & Preprocessing Metrics")
    pdf.set_font("Helvetica", "", 8)
    if not trials:
        pdf.multi_cell(0, 5, "No experimental trials recorded.", new_x="LMARGIN", new_y="NEXT")
    else:
        for t in trials:
            pdf.set_font("Helvetica", "B", 8)
            summary = (
                f"Trial #{t.get('Trial #', '—')}: {t.get('Configuration', '—')} | "
                f"Tokens: {t.get('Input Tokens', 0)} -> {t.get('Final Tokens', 0)} | "
                f"Vocab: {t.get('Vocabulary', 0)} | Reduction: {float(t.get('Reduction (%)', 0)):.2f}%"
            )
            pdf.multi_cell(0, 4.5, pdf_safe(summary), new_x="LMARGIN", new_y="NEXT")
            if "Boolean Query" in t:
                pdf.set_font("Helvetica", "I", 7.5)
                b_info = f"  Query: '{t.get('Boolean Query', '')}' => Matches: {t.get('Matches', 'None')}"
                pdf.multi_cell(0, 4, pdf_safe(b_info), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, "Graph 1: Input vs Final Tokens Across Trials", new_x="LMARGIN", new_y="NEXT")
        draw_pdf_tokens_chart(pdf, trials)

        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, "Graph 2: Vocabulary Reduction (%) Across Trials", new_x="LMARGIN", new_y="NEXT")
        draw_pdf_reduction_chart(pdf, trials)

    # 4. Boolean Retrieval Evaluation
    pdf_section_title(pdf, 4, "Boolean Retrieval Model Evaluation")
    pdf.set_font("Helvetica", "", 8.5)
    if latest_query:
        pdf.multi_cell(0, 4.5, pdf_safe(f"Evaluated Query: {latest_query}"), new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 4.5, pdf_safe(f"Matched Document Chunks: {latest_matches or 'None'}"), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.multi_cell(0, 4.5, "Boolean queries evaluated against preprocessed token sets (AND: intersection, OR: union, NOT: difference).", new_x="LMARGIN", new_y="NEXT")

    # 5. Comparative Analysis & Observations
    analysis_points, conclusion = build_trial_analysis(trials)
    pdf_section_title(pdf, 5, "Comparative Analysis & Observations")
    pdf.set_font("Helvetica", "", 8.5)
    for point in analysis_points:
        pdf.multi_cell(0, 4.5, pdf_safe(f"- {point}"), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 5, "Student Notes:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.multi_cell(0, 4.5, pdf_safe(notes.strip() or "No additional notes provided."), new_x="LMARGIN", new_y="NEXT")

    # 6. Conclusion
    pdf_section_title(pdf, 6, "Scientific Conclusion")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.multi_cell(0, 4.5, pdf_safe(conclusion), new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


def generate_certificate_pdf(
    student_name: str,
    roll_number: str,
    completion_date: str,
    quiz_score: int,
    total_quiz: int,
) -> bytes:
    """Generate a high-resolution landscape certificate of completion."""
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    # Decorative border
    pdf.set_draw_color(30, 64, 175)
    pdf.set_line_width(2.5)
    pdf.rect(10, 10, 277, 190)

    pdf.set_draw_color(245, 158, 11)
    pdf.set_line_width(0.8)
    pdf.rect(13, 13, 271, 184)

    # Header / Seal
    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 7, "INFORMATION RETRIEVAL VIRTUAL LABORATORY", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 64, 175)
    pdf.cell(0, 14, "CERTIFICATE OF COMPLETION", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 6, "This is to officially certify that", align="C", new_x="LMARGIN", new_y="NEXT")

    # Recipient Name
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(15, 23, 42)
    display_name = student_name.strip() if student_name.strip() else "Learner"
    pdf.cell(0, 10, pdf_safe(display_name), align="C", new_x="LMARGIN", new_y="NEXT")

    # Roll Number
    if roll_number.strip():
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, pdf_safe(f"Roll Number: {roll_number.strip()}"), align="C", new_x="LMARGIN", new_y="NEXT")

    # Course / Experiment Details
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(
        0, 6,
        "has successfully conducted the experimental simulation, comparative trials, and assessment for",
        align="C", new_x="LMARGIN", new_y="NEXT"
    )

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 64, 175)
    pdf.cell(0, 8, pdf_safe(EXPERIMENT["title"]), align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(
        0, 6,
        pdf_safe(f"Assessment Score: {quiz_score}/{total_quiz} ({(quiz_score/total_quiz*100):.0f}%) | Issued Date: {completion_date}"),
        align="C", new_x="LMARGIN", new_y="NEXT"
    )

    # Signatures
    pdf.set_y(155)
    pdf.set_draw_color(148, 163, 184)
    pdf.line(35, 170, 95, 170)
    pdf.line(200, 170, 260, 170)

    pdf.set_xy(35, 172)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(60, 4, "Virtual Lab Instructor", align="C")

    pdf.set_xy(200, 172)
    pdf.cell(60, 4, "Authorized Signature", align="C")

    pdf.set_xy(35, 177)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(60, 4, "Department of Computer Science & IR", align="C")

    pdf.set_xy(200, 177)
    pdf.cell(60, 4, "Virtual Laboratories Platform", align="C")

    return bytes(pdf.output())


def build_trial_analysis(trials: list[dict]) -> tuple[list[str], str]:
    if not trials:
        return (
            ["No experimental trials were recorded. Run at least two configurations to analyze."],
            "Text preprocessing and Boolean retrieval evaluation pending trial execution.",
        )

    best_red = max(trials, key=lambda t: float(t.get("Reduction (%)", 0)))
    min_voc = min(trials, key=lambda t: int(t.get("Vocabulary", 0)))
    configs = {str(t.get("Configuration", "")) for t in trials}

    analysis = [
        f"{len(trials)} trial(s) executed across {len(configs)} unique preprocessing configuration(s).",
        (
            f"Highest token reduction was {float(best_red.get('Reduction (%)', 0)):.2f}% in Trial #{best_red.get('Trial #', '—')}, "
            f"reducing {best_red.get('Input Tokens', 0)} raw tokens to {best_red.get('Final Tokens', 0)} standardized tokens."
        ),
        (
            f"Smallest vocabulary attained was {min_voc.get('Vocabulary', 0)} distinct terms in Trial #{min_voc.get('Trial #', '—')}, "
            "demonstrating effective morphological consolidation and stop-word filtering."
        ),
    ]

    conclusion = (
        "The preprocessing pipeline successfully cleaned, normalized, and reduced raw token streams. "
        "Morphological normalization (stemming/lemmatization) unified inflectional variants, and "
        "the Boolean Retrieval engine accurately executed exact set operations (AND, OR, NOT) over the preprocessed corpus."
    )
    return analysis, conclusion


def render_sidebar_logbook() -> None:
    """Render the Experimental Data Logbook persistently in the sidebar across all sections."""
    st.sidebar.divider()
    st.sidebar.subheader("Experimental Data Logbook")

    trials = st.session_state.get("trials", [])
    if trials:
        if st.sidebar.button("Clear Logged Trials", key="btn_clear_trials_sidebar", width="stretch"):
            st.session_state["trials"] = []
            st.rerun()

        trials_df = pd.DataFrame(trials)
        with st.sidebar.expander(f"Logged Trials ({len(trials)})", expanded=True):
            st.dataframe(
                trials_df[["Trial #", "Input Tokens", "Final Tokens", "Reduction (%)"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "Trial #": st.column_config.NumberColumn("Trial #", alignment="center"),
                    "Input Tokens": st.column_config.NumberColumn("Input", alignment="center"),
                    "Final Tokens": st.column_config.NumberColumn("Final", alignment="center"),
                    "Reduction (%)": st.column_config.NumberColumn("Red (%)", format="%.1f%%", alignment="center"),
                },
            )
            st.download_button(
                "Download Trials as CSV",
                data=trials_df.to_csv(index=False).encode("utf-8"),
                file_name="preprocessing_trials.csv",
                mime="text/csv",
                key="btn_download_trials_sidebar",
                width="stretch",
            )
    else:
        st.sidebar.info("No trials recorded yet. Go to Simulation tab to record trials.")


# ======================================================================================
# 7. SECTION 1: PURPOSE
# ======================================================================================

def render_purpose_section() -> None:
    st.header("Purpose & Experiment Overview")
    st.info(f"**Experiment Aim:** {EXPERIMENT['aim']}")

    st.subheader("1. What is this Lab?")
    st.write(
        "This laboratory focuses on **Text Preprocessing, Normalization, and Boolean Retrieval**. "
        "It provides an interactive environment to clean, transform, and standardize raw human text into uniform tokens "
        "and evaluate set-based document retrieval using Boolean operators (`AND`, `OR`, `NOT`)."
    )

    st.subheader("2. Why do we perform Text Preprocessing & Normalization?")
    st.write(
        "Raw human text is noisy, messy, and filled with redundancies. Words vary in capitalization (`Text` vs `text`), "
        "contain punctuation (`apple!` vs `apple`), use conversational shortcuts (`don't` instead of `do not`), "
        "and include high-frequency filler words (`the`, `is`, `at`) that add noise without contributing unique meaning."
    )
    st.write(
        "Without uniform text preprocessing, information retrieval systems and search engines would treat identical words as completely separate terms. "
        "By applying systematic tokenization, case folding, stop-word filtering, and morphological reduction (stemming vs lemmatization), "
        "we create a clean vocabulary so queries match documents accurately and efficiently."
    )

    st.subheader("3. What do you Gain by Performing this Lab?")
    gains = [
        "**Practical Preprocessing Skills:** Understand how each stage (casing, punctuation, stop-words, stemming/lemmatization) cleans raw token streams.",
        "**Vocabulary Compression Insights:** Observe how text normalization reduces overall token count and vocabulary size, improving storage and search speed.",
        "**Stemming vs Lemmatization Comparison:** Learn the practical trade-offs between mechanical suffix stripping (stemming) and dictionary base form mapping (lemmatization).",
        "**Boolean Retrieval Evaluation:** Master exact set theory operations (`AND`, `OR`, `NOT`) to retrieve relevant matching document chunks.",
    ]
    for gain in gains:
        st.markdown(f"- {gain}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prerequisites")
        for prereq in EXPERIMENT["prerequisites"]:
            st.markdown(f"- {prereq}")

    with col2:
        st.subheader("Expected Outcome")
        st.success(EXPERIMENT["expected_outcome"])


# ======================================================================================
# 8. SECTION 2: THEORY
# ======================================================================================

def render_theory_section() -> None:
    st.header("Theoretical Framework & Concepts")
    st.info(f"**Aim:** {EXPERIMENT['aim']}")

    st.subheader("1. What is Text Preprocessing?")
    st.markdown(
        f"Imagine reading a document where words have mixed capitalization, attached punctuation, or informal contractions. "
        f"Search engines require clean text to analyze documents accurately. "
        f"<b>Text Preprocessing</b> is the step-by-step cleaning line that transforms raw text into standardized units called "
        f"{key_term_html('Tokenization', 'tokens')}.",
        unsafe_allow_html=True,
    )

    st.subheader("2. Core Preprocessing Stages Explained")

    st.markdown(f"### **{key_term_html('Case Folding')} (Lowercasing)**", unsafe_allow_html=True)
    st.write(
        "Computers treat uppercase and lowercase characters as separate symbols. "
        "Case folding converts all text to lowercase so that **Information**, **INFORMATION**, and **information** match as the exact same word."
    )
    st.caption("**Example:** `Information Retrieval` ➔ `information retrieval`")

    st.markdown(f"### **{key_term_html('Contraction Expansion')}**", unsafe_allow_html=True)
    st.write(
        "Shortened words like *don't* or *can't* hide two distinct lexical components. "
        "Expanding contractions before stripping punctuation preserves underlying word meanings."
    )
    st.caption("**Example:** `don't keep noisy text` ➔ `do not keep noisy text`")

    st.markdown(f"### **{key_term_html('Noise Removal', 'Noise, Accent & Punctuation Removal')}**", unsafe_allow_html=True)
    st.write(
        "Special symbols, punctuation marks (commas, exclamation marks), and diacritical accents add extra clutter. "
        "Removing them leaves clean alphanumeric terms."
    )
    st.caption("**Example:** `Café!! fast-search` ➔ `cafe fast search`")

    st.markdown(f"### **{key_term_html('Tokenization')}**", unsafe_allow_html=True)
    st.write(
        "Tokenization breaks continuous sentences or token streams into individual units called tokens (words)."
    )
    st.caption("**Example:** `'search engines index text'` ➔ `['search', 'engines', 'index', 'text']`")

    st.markdown(f"### **{key_term_html('Stop-Word Filtering')}**", unsafe_allow_html=True)
    st.write(
        "Stop words are extremely common filler words—such as **the**, **is**, **at**, **and**, and **for**. "
        "Because these words appear everywhere, they provide little unique value for search and are filtered out."
    )
    st.caption("**Example:** `['the', 'retrieval', 'of', 'text']` ➔ `['retrieval', 'text']`")

    st.divider()
    st.subheader("3. Word Root Reduction: Stemming vs. Lemmatization")
    st.write(
        "Words naturally change their endings based on grammar (e.g., *study*, *studying*, *studied*, *studies*). "
        "To group all inflected variations under a single root concept, search systems use root reduction:"
    )

    st.markdown(f"#### **{key_term_html('Stemming')} (Fast Suffix Stripping)**", unsafe_allow_html=True)
    st.write(
        "Stemming applies simple rule-based heuristics to slice off word endings (suffixes). "
        "It is fast, but because it follows mechanical rules, the resulting stem might not always be a real dictionary word."
    )
    st.caption("**Example:** `studying`, `studied` ➔ stem `studi` | `organize`, `organization` ➔ stem `organ`")

    st.markdown(f"#### **{key_term_html('Lemmatization')} (Dictionary-Based Base Form)**", unsafe_allow_html=True)
    st.write(
        "Lemmatization uses full vocabulary dictionaries and grammatical analysis to find the true dictionary base form (the *lemma*). "
        "It always produces valid, meaningful words."
    )
    st.caption("**Example:** `studying`, `studied` ➔ lemma `study` | `were`, `was`, `been` ➔ lemma `be` | `feet` ➔ lemma `foot`")

    st.divider()
    st.subheader(f"4. Finding Documents: The {key_term_html('Boolean Retrieval')} Model")
    st.write(
        "Once documents and queries are preprocessed into standardized tokens, the search engine applies "
        "simple set logic (**Boolean operators**) to locate exact document matches:"
    )

    st.markdown(
        f"""
        - **{key_term_html('AND Operator', 'AND (Intersection)')}:** Returns documents containing **both** terms.  
          *Example:* `information AND retrieval` matches documents containing both *information* and *retrieval*.
        - **{key_term_html('OR Operator', 'OR (Union)')}:** Returns documents containing **at least one** of the terms.  
          *Example:* `text OR search` matches documents containing either *text*, *search*, or both.
        - **{key_term_html('NOT Operator', 'NOT (Difference)')}:** Excludes documents containing an unwanted term.  
          *Example:* `retrieval AND NOT noisy` matches documents with *retrieval* while excluding any that mention *noisy*.
        - **Parentheses ( ) :** Enforces grouping and execution order in complex queries.  
          *Example:* `(system OR process) AND index` evaluates the OR query first, then intersects with *index*.
        """,
        unsafe_allow_html=True,
    )


# ======================================================================================
# 9. SECTION 3: SIMULATION & BOOLEAN RETRIEVAL SANDBOX
# ======================================================================================

def render_simulation_section() -> None:
    st.header("Interactive Preprocessing & Boolean Retrieval Sandbox")
    st.info("Input chunked raw tokens, configure preprocessing stages, and test Boolean retrieval expressions.")

    source_mode = st.radio("Token Input Source", ["Use Default Token Stream", "Enter / Paste Tokens", "Upload TXT / CSV"], horizontal=True)
    raw_tokens_input = ""

    if source_mode == "Use Default Token Stream":
        raw_tokens_input = DEFAULT_TOKENS
    elif source_mode == "Enter / Paste Tokens":
        raw_tokens_input = st.text_area(
            "Raw Token Stream (comma-separated or space-separated chunked tokens)",
            value=DEFAULT_TOKENS,
            height=120,
            help="Example: my,name,is,x,i,am,a,student,studying,retrieval,systems",
        )
    else:
        uploaded_file = st.file_uploader("Upload CSV or TXT file containing tokens", type=["txt", "csv"])
        if uploaded_file:
            raw_tokens_input = read_uploaded_file(uploaded_file)
            if not raw_tokens_input:
                st.warning("Could not extract tokens from uploaded file.")
        else:
            raw_tokens_input = DEFAULT_TOKENS

    if not raw_tokens_input.strip():
        st.warning("Please enter or upload token input to proceed.")
        return

    st.subheader("Pipeline Configuration Controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        lowercase = st.checkbox("Convert to lowercase (Case Folding)", value=True)
        expand_contractions = st.checkbox("Expand contractions", value=True)
        remove_accents = st.checkbox("Remove accents / diacritics", value=True)
    with c2:
        remove_punctuation = st.checkbox("Remove punctuation", value=True)
        remove_numbers = st.checkbox("Remove numbers", value=False)
        remove_stopwords = st.checkbox("Remove stop words", value=True)
    with c3:
        morphology = st.radio("Word Morphology", ["Lemmatization", "Stemming", "None"])

    options = PipelineOptions(
        lowercase=lowercase,
        expand_contractions=expand_contractions,
        remove_accents=remove_accents,
        remove_punctuation=remove_punctuation,
        remove_numbers=remove_numbers,
        remove_stopwords=remove_stopwords,
        morphology=morphology,
    )

    result = run_pipeline(raw_tokens_input, options)
    st.plotly_chart(make_pipeline_flow_figure(morphology), width="stretch", config={"displayModeBar": False})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Input Tokens", result.original_token_count)
    m2.metric("Final Processed Tokens", result.final_token_count)
    m3.metric("Vocabulary Size", result.vocabulary_size)
    m4.metric("Token Reduction", f"{result.reduction_percent:.1f}%")

    st.subheader("Stage-by-Stage Transformation Output")
    stages = [
        ("Raw Input Tokens", raw_tokens_input),
        ("Normalized Stream", result.normalized),
        ("Tokenization (Tokens)", " | ".join(result.tokens)),
        ("Stop-Word Filtering", " | ".join(result.filtered_tokens)),
        (f"Morphology ({morphology})", " | ".join(result.final_tokens)),
        ("Clean Standardized Corpus", result.output_text),
    ]

    html_rows = []
    for stage_name, stage_val in stages:
        escaped_val = stage_val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html_rows.append(
            f"""
            <tr style="border-bottom: 1px solid rgba(128,128,128,0.15);">
                <td style="padding: 10px 12px; font-weight: 600; text-align: center; vertical-align: middle; width: 28%; white-space: nowrap;">
                    {stage_name}
                </td>
                <td style="padding: 6px 10px; width: 72%; text-align: left;">
                    <div style="overflow-x: auto; white-space: nowrap; max-width: 100%; padding: 8px 12px; background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.2); border-radius: 6px; font-family: monospace; font-size: 0.92rem; color: var(--text-color);">
                        {escaped_val}
                    </div>
                </td>
            </tr>
            """
        )

    table_html = f"""
    <div style="width: 100%; max-width: 100%; border: 1px solid rgba(128,128,128,0.2); border-radius: 8px; overflow: hidden; margin-bottom: 1rem;">
        <table style="width: 100%; border-collapse: collapse; margin: 0;">
            <thead>
                <tr style="background-color: var(--secondary-background-color); border-bottom: 2px solid rgba(128,128,128,0.25);">
                    <th style="padding: 10px 12px; text-align: center; width: 28%;">Stage / Operation</th>
                    <th style="padding: 10px 12px; text-align: center; width: 72%;">Sample Output (Scroll horizontally inside box)</th>
                </tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    if result.removed_stopwords:
        st.caption(f"Filtered Stop Words ({len(result.removed_stopwords)}): {', '.join(sorted(set(result.removed_stopwords)))}")

    if result.frequencies:
        st.subheader("Processed Token Frequency Distribution")
        top_freq = pd.DataFrame(list(result.frequencies.items())[:15], columns=["Token", "Frequency"])
        bar_colors = [
            "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
            "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
            "#14b8a6", "#d946ef", "#eab308", "#38bdf8", "#a855f7",
        ]
        fig_freq = go.Figure(
            go.Bar(
                x=top_freq["Token"],
                y=top_freq["Frequency"],
                marker_color=bar_colors[: len(top_freq)],
                text=top_freq["Frequency"],
                textposition="outside",
            )
        )
        fig_freq.update_layout(
            xaxis_title="Token",
            yaxis_title="Count",
            height=340,
            margin={"l": 20, "r": 20, "t": 30, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_freq, width="stretch")

    st.divider()

    # Boolean Retrieval Engine
    st.subheader("Boolean Retrieval Engine")
    st.write(
        "Evaluate compound Boolean queries (`AND`, `OR`, `NOT`, and `( )`) over the preprocessed corpus. "
        "The corpus is partitioned into chunked documents ($D_1, D_2, \\dots$) to evaluate multi-document retrieval."
    )

    doc_chunks = split_into_chunks(raw_tokens_input, chunk_size=8)
    if not doc_chunks:
        doc_chunks = [raw_tokens_input]

    with st.expander(f"Inspect Corpus Document Chunks ({len(doc_chunks)} Chunked Documents)", expanded=False):
        chunk_rows = []
        for d_id, chunk in enumerate(doc_chunks, start=1):
            p_chunk = run_pipeline(chunk, options)
            chunk_rows.append({
                "Document": f"D{d_id}",
                "Raw Chunk": chunk,
                "Preprocessed Tokens": ", ".join(p_chunk.final_tokens),
            })
        st.dataframe(pd.DataFrame(chunk_rows), hide_index=True, width="stretch")

    st.write("**Quick Sample Queries:**")
    sq1, sq2, sq3, sq4 = st.columns(4)
    if sq1.button("information AND retrieval"):
        st.session_state["b_query"] = "information AND retrieval"
    if sq2.button("text OR search"):
        st.session_state["b_query"] = "text OR search"
    if sq3.button("retrieval AND NOT noisy"):
        st.session_state["b_query"] = "retrieval AND NOT noisy"
    if sq4.button("(system OR process) AND index"):
        st.session_state["b_query"] = "(system OR process) AND index"

    if "b_query" not in st.session_state:
        st.session_state["b_query"] = "information AND retrieval"

    boolean_query = st.text_input(
        "Enter Boolean Query (supports AND, OR, NOT, parentheses):",
        value=st.session_state["b_query"],
    )
    st.session_state["b_query"] = boolean_query

    matched_docs, logic_traces = evaluate_boolean_query(boolean_query, doc_chunks, options)
    matched_doc_str = ", ".join(f"D{d}" for d in sorted(list(matched_docs))) if matched_docs else "No matching documents"

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Matching Documents", len(matched_docs))
        st.success(f"**Matched:** {matched_doc_str}")

    with res_col2:
        st.write("**Step-by-Step Boolean Logic Trace:**")
        for trace in logic_traces:
            st.code(trace, language="text")

    if matched_docs:
        matched_details = []
        for d_id in sorted(list(matched_docs)):
            chunk_text = doc_chunks[d_id - 1]
            p_chunk = run_pipeline(chunk_text, options)
            matched_details.append({
                "Document": f"D{d_id}",
                "Raw Tokens": chunk_text,
                "Matched Preprocessed Tokens": ", ".join(p_chunk.final_tokens),
            })
        st.dataframe(pd.DataFrame(matched_details), hide_index=True, width="stretch")

    st.download_button(
        "Download Clean Standardized Corpus (.txt)",
        data=result.output_text.encode("utf-8"),
        file_name="clean_preprocessed_tokens.txt",
        mime="text/plain",
        width="stretch",
    )

    # Record Current Trial to Logbook
    st.divider()
    st.subheader("Record Experimental Trial")
    st.markdown('<div class="record-trial-btn">', unsafe_allow_html=True)
    if st.button("Record Current Trial to Logbook", width="stretch"):
        trial_record = {
            "Trial #": len(st.session_state["trials"]) + 1,
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Configuration": configuration_label(options),
            "Input Tokens": result.original_token_count,
            "Final Tokens": result.final_token_count,
            "Vocabulary": result.vocabulary_size,
            "Reduction (%)": result.reduction_percent,
            "Boolean Query": boolean_query,
            "Matches": matched_doc_str,
            "Clean Corpus": result.output_text,
        }
        st.session_state["trials"].append(trial_record)
        st.success(f"Trial #{trial_record['Trial #']} successfully logged into the sidebar logbook!")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================================================
# 10. SECTION 4: QUIZ
# ======================================================================================

def render_quiz_section() -> None:
    st.header("Conceptual Assessment Quiz")
    st.write("Test your understanding of text preprocessing, normalization, and Boolean retrieval.")

    if st.session_state["quiz_question_set"] is None:
        st.session_state["quiz_question_set"] = random.sample(
            QUIZ_QUESTIONS, min(10, len(QUIZ_QUESTIONS))
        )

    if st.button("Generate New Random Quiz"):
        st.session_state["quiz_question_set"] = random.sample(
            QUIZ_QUESTIONS, min(10, len(QUIZ_QUESTIONS))
        )
        st.session_state["quiz_answers"] = {}
        st.session_state["quiz_submitted"] = False
        st.session_state["quiz_score"] = 0
        for q in QUIZ_QUESTIONS:
            st.session_state.pop(f"quiz_{q['id']}", None)

    quiz_set = st.session_state["quiz_question_set"]
    st.caption(f"Question Bank: {len(QUIZ_QUESTIONS)} questions | Current Assessment: {len(quiz_set)} questions")

    with st.form("quiz_form"):
        responses = {}
        for idx, q in enumerate(quiz_set, start=1):
            st.markdown(f"**Question {idx}:** {q['question']}")
            selected = st.radio(
                f"Options for Q{idx}",
                q["options"],
                index=None,
                key=f"quiz_{q['id']}",
                label_visibility="collapsed",
            )
            responses[q["id"]] = q["options"].index(selected) if selected is not None else None
            st.write("")

        submitted = st.form_submit_button("Submit Quiz for Grading", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Quiz Results & Detailed Explanations")
        for idx, q in enumerate(quiz_set, start=1):
            resp = responses[q["id"]]
            if resp == q["answer"]:
                score += 1
                st.success(f"**Q{idx}: Correct!** {q['explanation']}")
            elif resp is None:
                st.warning(f"**Q{idx}: Unanswered.** Correct answer: {q['options'][q['answer']]}. {q['explanation']}")
            else:
                st.error(f"**Q{idx}: Incorrect.** Your answer: {q['options'][resp]} | Correct: {q['options'][q['answer']]}. {q['explanation']}")

        st.session_state["quiz_score"] = score
        pct = (score / len(quiz_set)) * 100
        st.info(f"Final Score: **{score}/{len(quiz_set)} ({pct:.0f}%)**")

    elif st.session_state["quiz_submitted"]:
        st.info(f"Latest Recorded Score: **{st.session_state['quiz_score']}/{len(quiz_set)}**")


# ======================================================================================
# 11. SECTION 5: REPORT GENERATION
# ======================================================================================

def render_report_section() -> None:
    st.header("Laboratory Report Generation")
    st.write(
        "Review your experimental results, comparative charts, automatic scientific analysis, "
        "and generate an official downloadable PDF lab report with embedded charts."
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        s_name = st.text_input("Student Name", value=st.session_state["student_info"]["name"])
    with r2:
        s_roll = st.text_input("Roll Number", value=st.session_state["student_info"]["roll"])
    with r3:
        s_date = st.date_input("Experiment Date", value=date.today())

    st.session_state["student_info"] = {"name": s_name, "roll": s_roll, "date": str(s_date)}

    def_notes = (
        "I observed that stop-word removal and case folding significantly reduced the token count and vocabulary size. "
        "Lemmatization preserved valid dictionary roots while stemming aggressively removed affixes. "
        "The Boolean Retrieval engine accurately retrieved document chunks using exact set operations."
    )
    student_notes = st.text_area(
        "Student Observations & Inferences",
        value=st.session_state["student_notes"] or def_notes,
        height=120,
    )
    st.session_state["student_notes"] = student_notes

    trials = st.session_state["trials"]
    total_q = len(st.session_state["quiz_question_set"]) if st.session_state.get("quiz_question_set") else 10
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Recorded Trials", len(trials))
    q_str = f"{st.session_state['quiz_score']}/{total_q}" if st.session_state["quiz_submitted"] else "Pending"
    m_col2.metric("Quiz Assessment Score", q_str)

    if not trials:
        st.warning("Please record at least 2 simulation trials in the Simulation tab to generate comparison graphs.")
    else:
        st.subheader("Logged Experimental Trials")
        st.dataframe(pd.DataFrame(trials), hide_index=True, width="stretch")

        st.subheader("Comparative Visualizations")
        labels = [f"Trial {t.get('Trial #', i + 1)}" for i, t in enumerate(trials)]

        fig_tokens = go.Figure()
        fig_tokens.add_trace(go.Bar(name="Input Tokens", x=labels, y=[t.get("Input Tokens", 0) for t in trials], marker_color="#3b82f6"))
        fig_tokens.add_trace(go.Bar(name="Final Tokens", x=labels, y=[t.get("Final Tokens", 0) for t in trials], marker_color="#10b981"))
        fig_tokens.update_layout(
            title="Input vs Final Processed Tokens Across Trials",
            xaxis_title="Trial",
            yaxis_title="Token Count",
            barmode="group",
            height=340,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_tokens, width="stretch")

        red_colors = ["#8b5cf6", "#ec4899", "#06b6d4", "#f59e0b", "#10b981", "#ef4444", "#3b82f6"]
        fig_red = go.Figure(
            go.Bar(
                x=labels,
                y=[t.get("Reduction (%)", 0) for t in trials],
                marker_color=red_colors[: len(trials)],
                text=[f"{float(t.get('Reduction (%)', 0)):.1f}%" for t in trials],
                textposition="outside",
            )
        )
        fig_red.update_layout(
            title="Corpus Token Reduction Percentage Achieved",
            xaxis_title="Trial",
            yaxis_title="Reduction (%)",
            height=320,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_red, width="stretch")

    analysis_points, conclusion = build_trial_analysis(trials)
    st.subheader("Automated Comparative Analysis Preview")
    for pt in analysis_points:
        st.markdown(f"- {pt}")

    st.subheader("Conclusion Preview")
    st.success(conclusion)

    latest_q = trials[-1].get("Boolean Query", "") if trials else st.session_state.get("b_query", "")
    latest_m = trials[-1].get("Matches", "") if trials else ""

    pdf_bytes = generate_pdf_report(
        student_name=s_name,
        roll_number=s_roll,
        experiment_date=str(s_date),
        trials=trials,
        notes=student_notes,
        quiz_score=st.session_state["quiz_score"],
        total_quiz_questions=total_q,
        latest_query=latest_q,
        latest_matches=latest_m,
    )
    st.download_button(
        "Download Official Laboratory Report (PDF)",
        data=pdf_bytes,
        file_name="preprocessing_boolean_retrieval_lab_report.pdf",
        mime="application/pdf",
        type="primary",
        width="stretch",
    )

    session_dict = {
        "experiment": EXPERIMENT,
        "student": st.session_state["student_info"],
        "trials": trials,
        "quiz_score": st.session_state["quiz_score"],
        "student_notes": student_notes,
        "analysis": analysis_points,
        "conclusion": conclusion,
    }
    st.download_button(
        "Download Session Data (.json)",
        data=json.dumps(session_dict, indent=2).encode("utf-8"),
        file_name="session_data.json",
        mime="application/json",
        width="stretch",
    )


# ======================================================================================
# 12. SECTION 6: CERTIFICATE
# ======================================================================================

def render_certificate_section() -> None:
    st.header("Certificate of Completion")
    st.write(
        "Earn and download your official Certificate of Completion upon finishing the experimental "
        "simulation trials and assessment quiz."
    )

    s_name = st.session_state["student_info"]["name"]
    s_roll = st.session_state["student_info"]["roll"]
    s_date = st.session_state["student_info"]["date"]
    quiz_submitted = st.session_state["quiz_submitted"]
    quiz_score = st.session_state["quiz_score"]
    total_q = len(st.session_state["quiz_question_set"]) if st.session_state["quiz_question_set"] else 10

    c1, c2 = st.columns(2)
    with c1:
        s_name_input = st.text_input("Name on Certificate", value=s_name or "Alex Student")
        st.session_state["student_info"]["name"] = s_name_input
    with c2:
        s_roll_input = st.text_input("Roll / Registration ID", value=s_roll or "CS-2026-001")
        st.session_state["student_info"]["roll"] = s_roll_input

    # Certificate Preview Card
    st.subheader("Certificate Preview")
    cert_container = st.container(border=True)
    with cert_container:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 25px; border: 2px solid #3b82f6; border-radius: 8px; background-color: var(--secondary-background-color); color: var(--text-color);">
                <div style="font-size: 0.9rem; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.75;">Information Retrieval Virtual Laboratory</div>
                <h2 style="margin: 8px 0; font-family: serif; color: #3b82f6;">Certificate of Completion</h2>
                <p style="font-style: italic; margin-top: 12px; opacity: 0.85;">This is proudly presented to</p>
                <h3 style="margin: 10px 0; text-decoration: underline;">{s_name_input}</h3>
                <p style="margin-bottom: 12px; opacity: 0.75;">Roll Number / ID: <b>{s_roll_input}</b></p>
                <p style="font-size: 15px; max-width: 650px; margin: 0 auto; opacity: 0.9;">
                    for successfully demonstrating proficiency in text normalization, tokenization, stop-word removal, 
                    morphological stemming vs lemmatization, and Boolean Retrieval set operations in <b>{EXPERIMENT['title']}</b>.
                </p>
                <div style="margin-top: 20px; display: flex; justify-content: space-around; opacity: 0.85;">
                    <div><b>Date:</b> {s_date}</div>
                    <div><b>Assessment Score:</b> {quiz_score}/{total_q} ({((quiz_score/total_q)*100):.0f}%)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    cert_pdf = generate_certificate_pdf(
        student_name=s_name_input,
        roll_number=s_roll_input,
        completion_date=str(s_date),
        quiz_score=quiz_score,
        total_quiz=total_q,
    )

    st.download_button(
        "Download Official Certificate (PDF)",
        data=cert_pdf,
        file_name=f"certificate_{s_name_input.replace(' ', '_').lower()}.pdf",
        mime="application/pdf",
        type="primary",
        width="stretch",
    )


# ======================================================================================
# 13. SECTION 7: REFERENCES
# ======================================================================================

def render_references_section() -> None:
    st.header("Academic References & Reading Material")
    st.write(
        "The theoretical foundation, algorithms, and experimental pedagogy implemented in this laboratory "
        "are drawn from foundational texts in Information Retrieval, Computational Linguistics, and Open Educational Platforms:"
    )

    references = [
        {
            "title": "Introduction to Information Retrieval",
            "authors": "Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze",
            "publisher": "Cambridge University Press, 2008",
            "relevance": "Chapters 1 & 2: Boolean Retrieval, Inverted Index Construction, Tokenization, and Stemming/Lemmatization.",
            "url": "https://nlp.stanford.edu/IR-book/",
        },
        {
            "title": "Speech and Language Processing (3rd ed. draft)",
            "authors": "Daniel Jurafsky and James H. Martin",
            "publisher": "Pearson / Stanford University, 2024",
            "relevance": "Chapter 2: Text Normalization, Regular Expressions, Word Tokenization, and Edit Distance.",
            "url": "https://web.stanford.edu/~jurafsky/slp3/",
        },
        {
            "title": "An Algorithm for Suffix Stripping",
            "authors": "Martin F. Porter",
            "publisher": "Program: Electronic Library and Information Systems, 14(3): 130–137, 1980",
            "relevance": "Original formal rules and algorithmic design for the Porter Stemmer.",
            "url": "https://tartarus.org/martin/PorterStemmer/",
        },
        {
            "title": "Virtual Labs Platform",
            "authors": "Ministry of Education (Govt. of India) & IIT Kharagpur",
            "publisher": "National Mission on Education through ICT",
            "relevance": "Standardized 7-section pedagogy, self-evaluation paradigms, and virtual simulation benchmarks.",
            "url": "https://www.vlabs.ac.in/",
        },
    ]

    for idx, ref in enumerate(references, start=1):
        with st.container(border=True):
            st.subheader(ref["title"])
            st.markdown(f"**Authors:** {ref['authors']}")
            st.markdown(f"**Publisher / Source:** {ref['publisher']}")
            st.markdown(f"**Topic Relevance:** {ref['relevance']}")
            st.markdown(f"[External Resource & Open Access Link]({ref['url']})")


# ======================================================================================
# 14. MAIN ENTRY POINT & NAVIGATION
# ======================================================================================

def main() -> None:
    st.set_page_config(
        page_title=EXPERIMENT["title"],
        layout="wide",
    )
    initialize_state()
    inject_custom_styles()

    section = st.sidebar.radio(
        "Lab Navigator",
        [
            "Purpose",
            "Theory",
            "Simulation",
            "Quiz",
            "Report Generation",
            "Certificate",
            "References",
        ],
    )
    st.sidebar.divider()
    st.sidebar.subheader("Progress Tracker")
    st.sidebar.write(f"- **Trials recorded:** {len(st.session_state['trials'])}")
    quiz_status = "Completed" if st.session_state["quiz_submitted"] else "Pending"
    st.sidebar.write(f"- **Quiz status:** {quiz_status}")
    if st.session_state["quiz_submitted"]:
        total_quiz_q = len(st.session_state["quiz_question_set"]) if st.session_state.get("quiz_question_set") else 10
        st.sidebar.write(f"- **Quiz score:** {st.session_state['quiz_score']}/{total_quiz_q}")

    # Render Experimental Data Logbook persistently in Sidebar across ALL sections
    render_sidebar_logbook()

    st.sidebar.divider()
    st.sidebar.caption(f"**Expected Outcome:** {EXPERIMENT['expected_outcome']}")

    if section == "Purpose":
        st.title(EXPERIMENT["title"])
        st.caption(
            f"Information Retrieval Virtual Laboratory · "
            f"Experiment {EXPERIMENT['number']} · Estimated time: {EXPERIMENT['estimated_minutes']}"
        )
        render_purpose_section()
    elif section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()
    elif section == "Certificate":
        render_certificate_section()
    elif section == "References":
        render_references_section()


if __name__ == "__main__":
    main()
