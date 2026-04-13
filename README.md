# Customer Service Log Analysis with T5 and Zero-Shot Classification

An NLP pipeline that processes raw customer service call logs. It extracts structured fields, summarises each conversation using Google FLAN-T5, detects cancellation requests via zero-shot classification, extracts the reasons behind cancellations, and visualises common terminology with word clouds.

## Business Context

Manually reviewing large volumes of customer service transcripts is time-consuming. This pipeline automates three high-value tasks: conversation summarisation, cancellation detection, and root cause extraction, enabling a customer service team to quickly identify at-risk customers and recurring issues.

## Dataset

`Call_Logs.csv` contains a `Logs` column where each entry is a newline-delimited string with date, time, and conversation content.

## Methodology

**Log Parsing:** Each raw log entry is split into Date, Time, and Conversation fields.

**Summarisation:** Google FLAN-T5 Base (`google/flan-t5-base`) generates a concise summary for each conversation by answering the prompt "What were the main points in that conversation?"

**Cancellation Detection:** A zero-shot classifier (`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`) classifies each conversation as `cancellation` or `other` without any task-specific fine-tuning. The multilingual model supports customer service contexts where conversations may be in different languages.

**Reason Extraction:** For conversations flagged as cancellations, FLAN-T5 is prompted to identify the issues that led the customer to cancel.

**Word Clouds:** Separate word clouds for raw conversations and generated summaries visualise the most frequent terms.

## Project Structure

```
12_customer_service_nlp/
├── customer_service_nlp.py  # Full pipeline
├── requirements.txt
└── README.md
```

## Requirements

```
pandas
matplotlib
transformers
torch
wordcloud
sentencepiece
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Place `Call_Logs.csv` in the same directory and run:

```bash
python customer_service_nlp.py
```

Outputs: `wordcloud_conversations.png`, `wordcloud_summaries.png`, and printed summaries and cancellation reasons.

## Notes

Model inference runs on CPU by default. On large datasets, GPU acceleration via `device=0` in the pipeline call will significantly reduce runtime. Both FLAN-T5 and the zero-shot classifier are downloaded automatically on first run from Hugging Face Hub.
