from transformers import pipeline

class AnswerGenerator:
    def __init__(self, model_name="gpt2"):
        self.generator = pipeline("text-generation", model=model_name)

    def generate_answer(self, context, query, max_length=100):
        """Generate an answer using the retrieved context and query."""
        input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        output = self.generator(input_text, max_length=max_length, num_return_sequences=1)
        return output[0]["generated_text"]