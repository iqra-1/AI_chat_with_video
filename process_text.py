# # def process_extracted_text(extracted_text):
# #     # For now, just return the text, but this can be expanded to summaries or specific text processing
# #     return extracted_text


# from transformers import pipeline


# def summarize_transcript(transcript):
#     print("Summary")
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
#     summary = summarizer(transcript, max_length=200, min_length=50, do_sample=False)
#     return summary[0]["summary_text"]


# def process_extracted_text(extracted_text):
#     summary = summarize_transcript(extracted_text)
#     return summary
from transformers import pipeline


def summarize_transcript(transcript):
    print("Starting summarization...")

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

    # Split the transcript into manageable chunks
    max_chunk_size = 1024  # Model's token limit
    chunks = [
        transcript[i : i + max_chunk_size]
        for i in range(0, len(transcript), max_chunk_size)
    ]

    # Summarize each chunk and combine results
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
        summaries.append(summary[0]["summary_text"])

    # Combine all summaries into a final result
    final_summary = " ".join(summaries)
    return final_summary


def process_extracted_text(extracted_text):
    summary = summarize_transcript(extracted_text)
    return summary
