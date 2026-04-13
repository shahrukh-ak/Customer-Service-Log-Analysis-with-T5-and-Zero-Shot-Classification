"""
Customer Service Log Analysis with T5 and Zero-Shot Classification
===================================================================
Processes raw customer service call logs to extract structured fields,
summarises conversations with Google FLAN-T5, classifies cancellation
requests using zero-shot classification, extracts cancellation reasons,
and visualises common topics via word clouds.

Dataset: Call_Logs.csv
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    pipeline,
)
warnings.filterwarnings("ignore")


# ── Data Loading and Parsing ──────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load call log CSV and display first row."""
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    return df


def extract_log_fields(row: pd.Series) -> tuple:
    """
    Parse a raw log entry into (date, time, conversation).
    Expects the Logs field to be newline-delimited with
    'Date: ...', 'Time: ...', followed by conversation lines.
    """
    lines = row["Logs"].split("\n")
    date  = lines[0].split(": ")[1]
    time  = lines[1].split(": ")[1]
    conv  = "\n".join(line for line in lines[3:] if line.strip())
    return date, time, conv


def parse_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structured fields from raw Logs column and drop originals."""
    df[["Date", "Time", "Conversation"]] = df.apply(
        extract_log_fields, axis=1, result_type="expand"
    )
    df = df.drop(columns=["Logs", "Unnamed: 0"], errors="ignore")
    return df


# ── Conversation Summarisation ────────────────────────────────────────────────

def load_t5_model(model_name: str = "google/flan-t5-base"):
    """Load FLAN-T5 tokeniser and model."""
    tokeniser = T5Tokenizer.from_pretrained(model_name)
    model     = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokeniser, model


def summarise_conversations(df: pd.DataFrame, tokeniser, model) -> pd.DataFrame:
    """
    Generate a one-sentence summary for each conversation using FLAN-T5
    and append results as a new 'Summary' column.
    """
    summaries = []
    for conv in df["Conversation"]:
        prompt = f"{conv}\n\nWhat were the main points in that conversation?"
        input_ids  = tokeniser(prompt, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids)
        summary    = tokeniser.decode(output_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    df["Summary"] = summaries
    return df


# ── Cancellation Detection ────────────────────────────────────────────────────

def detect_cancellations(df: pd.DataFrame,
                          model_name: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
                          ) -> pd.DataFrame:
    """
    Use zero-shot classification to flag conversations that involve
    a cancellation request.
    """
    labels     = ["cancellation", "other"]
    classifier = pipeline("zero-shot-classification", model=model_name)

    results = [
        classifier(conv, labels)["labels"][0]
        for conv in df["Conversation"]
    ]
    df["Cancellation"] = [r == "cancellation" for r in results]
    print(f"Cancellation conversations detected: {df['Cancellation'].sum()}")
    return df


# ── Cancellation Reason Extraction ───────────────────────────────────────────

def extract_cancellation_reasons(df: pd.DataFrame, tokeniser, model) -> pd.DataFrame:
    """
    For conversations flagged as cancellations, use FLAN-T5 to extract
    the reasons given. Non-cancellation rows receive 'None'.
    """
    def get_reason(row: pd.Series) -> str:
        if not row["Cancellation"]:
            return "None"
        prompt = (
            f"{row['Conversation']}\n\n"
            "What are the issues that led the client to cancel their subscription?"
        )
        input_ids  = tokeniser(prompt, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids)
        return tokeniser.decode(output_ids[0], skip_special_tokens=True)

    df["Cancellation_Reason"] = df.apply(get_reason, axis=1)

    print("\nCancellation Reasons:")
    for reason in df[df["Cancellation_Reason"] != "None"]["Cancellation_Reason"]:
        print(f"  - {reason}")

    return df


# ── Word Cloud Visualisation ──────────────────────────────────────────────────

def generate_wordcloud(text_series: pd.Series, title: str, output_file: str):
    """Generate and save a word cloud from a text series."""
    text = " ".join(text_series.dropna().tolist())
    wc   = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()
    print(f"Saved: {output_file}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "Call_Logs.csv"

    df = load_data(DATA_PATH)
    df = parse_logs(df)
    print(df.head())

    tokeniser, t5_model = load_t5_model()

    df = summarise_conversations(df, tokeniser, t5_model)

    df = detect_cancellations(df)

    df = extract_cancellation_reasons(df, tokeniser, t5_model)

    generate_wordcloud(df["Conversation"], "Conversations Word Cloud",  "wordcloud_conversations.png")
    generate_wordcloud(df["Summary"],      "Summaries Word Cloud",       "wordcloud_summaries.png")

    print("\nFinal DataFrame preview:")
    print(df.head())
