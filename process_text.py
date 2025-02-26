from transformers import pipeline

summarizer = pipeline("summarization", device="cpu")


def chunk_text(text, chunk_size=512):
    """Splits text into smaller chunks to fit model's token limit."""
    words = text.split()  # Simple split by words (consider using NLP-based chunking)
    return [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]


def process_extracted_text(extracted_text):
    chunks = chunk_text(extracted_text)
    summaries = [
        summarizer(chunk, max_length=1024, min_length=100, do_sample=False)[0][
            "summary_text"
        ]
        for chunk in chunks
    ]
    return " ".join(summaries)  # Combine summarized chunks
