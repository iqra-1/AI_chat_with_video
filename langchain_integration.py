import torch
from transformers import pipeline
import json
import re

# Initialize the text-generation pipeline with the model
model_id = "meta-llama/Llama-3.2-1B"
# HUGGINGFACE_TOKEN = "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"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device=0,
)


# def generate_answer_and_suggested_questions(context, question):
#     # Ensure entire context is sent (not truncated)
#     print("Full Context:", context)  # Debugging print

#     # Prompt for generating the answer
#     # answer_prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
#     answer_prompt = (
#         f"Context: {context}\n"
#         f"Question: {question}\n"
#         f"Provide a clear, concise, and complete answer to the question based on the context above."
#     )

#     print("Answer Generation Prompt:", answer_prompt)  # Debugging print

#     # Generate the answer using the model
#     with torch.no_grad():
#         answer_response = pipe(
#             answer_prompt,
#             max_new_tokens=150,
#             num_return_sequences=1,
#             top_k=50,
#             top_p=0.9,
#         )[0]["generated_text"]
#         print("Raw Answer Response:", answer_response)  # Debugging print

#         # Extract the answer by isolating text after "Answer:"
#         if "Answer:" in answer_response:
#             answer = answer_response.split("Answer:")[-1].strip()
#         else:
#             answer = answer_response.strip()
#         print("Extracted Answer:", answer)  # Debugging print

#         # Prompt for generating suggested questions, specifying array format for clarity
#         suggestion_prompt = f"Context: {context}\nQuestion: {question}\nPlease provide exactly three follow-up questions in an array format, each question as a distinct item related to the topic:"
#         print("Suggestion Generation Prompt:", suggestion_prompt)  # Debugging print

#         # Generate the suggested questions using the model
#         suggested_questions_response = pipe(
#             suggestion_prompt,
#             max_new_tokens=100,
#             num_return_sequences=1,
#             top_k=50,
#             top_p=0.9,
#         )[0]["generated_text"]
#         print(
#             "Raw Suggested Questions Response:", suggested_questions_response
#         )  # Debugging print

#         # Extract and clean the suggested questions
#         if (
#             "Please provide exactly three follow-up questions"
#             in suggested_questions_response
#         ):
#             suggested_questions_text = suggested_questions_response.split(
#                 "Please provide exactly three follow-up questions"
#             )[-1].strip()
#         else:
#             suggested_questions_text = suggested_questions_response.strip()

#         # Now split the text by punctuations like '.', '?' or '\n' (for distinct questions)
#         suggested_questions = re.split(
#             r"[?.!]\s*", suggested_questions_text
#         )  # Split by sentence-ending punctuation
#         suggested_questions = [
#             q.strip() for q in suggested_questions if q.strip()
#         ]  # Clean up any empty strings

#         # Ensure exactly 3 questions are extracted
#         suggested_questions = suggested_questions[:3]

#         # Remove any unwanted introductory phrase from the first question if present
#         if suggested_questions and "in an array format" in suggested_questions[0]:
#             suggested_questions[0] = re.sub(
#                 r"in an array format, each question as a distinct item related to the topic:\s*",
#                 "",
#                 suggested_questions[0],
#             )

#         print("Extracted Suggested Questions:", suggested_questions)  # Debugging print

#     # Build the response structure
#     response = {"answer": answer, "suggested_questions": suggested_questions}
#     print("Final Response:", json.dumps(response, indent=2))  # Debugging print

#     return response


# def generate_answer_and_suggested_questions(context, question):
#     # Prompt for generating the answer
#     answer_prompt = (
#         f"Context: {context}\n"
#         f"Question: {question}\n"
#         f"Provide a concise and complete answer to the question. Ensure the answer directly addresses the question without repeating unnecessary context.\n"
#         f"Answer:"
#     )

#     # Generate the answer
#     with torch.no_grad():
#         answer_response = pipe(
#             answer_prompt,
#             max_new_tokens=150,
#             num_return_sequences=1,
#             top_k=50,
#             top_p=0.9,
#         )[0]["generated_text"]

#         # Extract and clean the answer
#         if "Answer:" in answer_response:
#             answer = answer_response.split("Answer:")[-1].strip()
#         else:
#             answer = answer_response.strip()

#         # Validate and truncate answer
#         answer = answer[:500]
#         if len(answer.split()) < 5:
#             answer = "Unable to generate a valid answer based on the provided context."

#     # Prompt for generating suggested questions
#     suggestion_prompt = (
#         f"Context: {context}\n"
#         f"Question: {question}\n"
#         f"Generate exactly three relevant and distinct follow-up questions based on the context. "
#         f'Format the response as a JSON array, e.g., ["Question 1", "Question 2", "Question 3"].\n'
#         f"Suggested Questions:"
#     )

#     # Generate the suggested questions
#     with torch.no_grad():
#         suggested_questions_response = pipe(
#             suggestion_prompt,
#             max_new_tokens=100,
#             num_return_sequences=1,
#             top_k=50,
#             top_p=0.9,
#         )[0]["generated_text"]

#         # Extract and clean suggested questions
#         if "[" in suggested_questions_response and "]" in suggested_questions_response:
#             try:
#                 suggested_questions = json.loads(
#                     suggested_questions_response.split("Suggested Questions:")[
#                         -1
#                     ].strip()
#                 )
#             except json.JSONDecodeError:
#                 suggested_questions = []
#         else:
#             # Fallback: Extract potential questions from text
#             suggested_questions = re.split(r"[?.!]\s*", suggested_questions_response)
#             suggested_questions = [q.strip() for q in suggested_questions if q.strip()]

#         # Validate and ensure exactly 3 questions
#         if len(suggested_questions) < 3:
#             suggested_questions = [
#                 "What are the key takeaways from this context?",
#                 "How can this story inspire students to overcome challenges?",
#                 "What lessons can be learned from Ubaid's experiences?",
#             ][:3]

#     # Final response structure
#     response = {
#         "answer": answer,
#         "suggested_questions": suggested_questions,
#     }

#     print("Final Response:", json.dumps(response, indent=2))  # Debugging print
#     return response


def generate_answer_and_suggested_questions(context, question):
    import re

    # Trim the context to avoid overloading the model
    trimmed_context = context[-2048:]  # Last 2048 characters

    # Generate Answer
    answer_prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Your task is to provide a clear and concise answer based ONLY on the given context. Avoid repeating the context verbatim. Focus on the main idea."
    )
    answer_response = pipe(
        answer_prompt, max_new_tokens=150, num_return_sequences=1, top_k=50, top_p=0.9
    )[0]["generated_text"]
    answer = (
        answer_response.split("Answer:")[-1].strip()
        if "Answer:" in answer_response
        else answer_response.strip()
    )
    print("Context:", context)
    print("Answer Prompt:", answer_prompt)
    print("Raw Answer Response:", answer_response)
    print("Final Answer:", answer)
    # Generate Suggested Questions
    suggestion_prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Your task is to generate exactly three creative and relevant follow-up questions. "
        f"Focus on deeper insights, applications, or areas for exploration related to the main idea."
    )
    suggested_questions_response = pipe(
        suggestion_prompt,
        max_new_tokens=100,
        num_return_sequences=1,
        top_k=50,
        top_p=0.9,
    )[0]["generated_text"]
    suggested_questions_text = suggested_questions_response.split(
        "Generate exactly three meaningful follow-up questions"
    )[-1].strip()
    suggested_questions = re.split(r"[?.!]\s*", suggested_questions_text)
    suggested_questions = [q.strip() for q in suggested_questions if q.strip()]

    print("Suggestion Prompt:", suggestion_prompt)
    print("Raw Suggested Questions Response:", suggested_questions_text)
    print("Final Suggested Questions:", suggested_questions)

    # Fallback for Empty or Incomplete Responses
    if not answer:
        answer = "Unable to generate an answer based on the provided context."
    if not suggested_questions or len(suggested_questions) < 3:
        suggested_questions = [
            "What are the main ideas discussed in this context?",
            "How can the concepts mentioned be applied in real life?",
            "What strategies can improve learning based on this context?",
        ]

    # Return Final Response
    response = {"answer": answer, "suggested_questions": suggested_questions[:3]}
    print("Final Response:", json.dumps(response, indent=2))
    return response


# Example usage
if __name__ == "__main__":
    example_context = "The Llama 3.2 collection of multilingual large language models is optimized for dialogue use cases."
    example_question = "What is the main purpose?"

    response = generate_answer_and_suggested_questions(
        example_context, example_question
    )
