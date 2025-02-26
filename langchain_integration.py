import torch
import re
from transformers import pipeline
from unsloth import FastLanguageModel

# Load model from Unsloth
model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # You can change the model if needed

# Load model using Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    load_in_4bit=True,  # Uses 4-bit quantization for efficiency
    max_seq_length=4096,  # Adjust as needed
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
# Prepare model for inference (REQUIRED for Unsloth)
FastLanguageModel.for_inference(model)

# Create text-generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=device)

# Fix tokenizer pad token issue
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id


def generate_answer_and_suggested_questions(context, question):
    # Improved answer generation
    answer_prompt = (
        f"### Instruction: Based on the following context, provide a clear and concise answer.\n"
        f"### Context: {context}\n"
        f"### Question: {question}\n"
        f"### Answer:"
    )

    answer_response = pipe(
        answer_prompt,
        max_new_tokens=150,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"]

    # Extract answer
    answer = answer_response.split("### Answer:")[-1].strip()
    print("answer inside a function", answer)
    answer = re.sub(r"\s+", " ", answer).strip()

    # Enhanced question suggestions
    suggestion_prompt = (
        f"### Instruction: Based on this context and question, suggest three relevant follow-up questions.\n"
        f"### Context: {context}\n"
        f"### Original Question: {question}\n"
        f"### Suggested Questions:\n"
        f"1."
    )

    suggested_questions_response = pipe(
        suggestion_prompt,
        max_new_tokens=120,
        num_return_sequences=1,
        temperature=0.8,
        top_k=40,
        do_sample=True,
    )[0]["generated_text"]

    # Extract questions using regex
    suggested_questions = re.findall(r"\d+\.\s*(.*?)\?", suggested_questions_response)
    print("suggested_questions inside a function", suggested_questions)
    suggested_questions = [q.strip() for q in suggested_questions[:3]]

    # Fallback if regex fails
    if not suggested_questions:
        suggested_questions = [
            "Can you explain this in more detail?",
            "What are the main factors to consider?",
            "How does this compare to similar concepts?",
        ]

    return {"answer": answer, "suggested_questions": suggested_questions}


# Example usage
if __name__ == "__main__":
    example_context = "The Llama 3.2 collection of multilingual large language models is optimized for dialogue use cases."
    example_question = "What is the main purpose?"

    response = generate_answer_and_suggested_questions(
        example_context, example_question
    )
    print(response["answer"])
    print("-" * 50)
    print("\n".join(response["suggested_questions"]))
