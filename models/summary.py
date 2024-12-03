import os
from ctransformers import AutoModelForCausalLM

class Summarizer:
    def __init__(self, model_path=r"D:\Minor\models\llama-2-7b-chat.ggmlv3.q2_K.bin"):
        """
        Initializes the summarizer using ctransformers with a quantized model.

        :param model_path: Path to the quantized LLaMA model file (GGML format).
        """
        # Load the quantized model
        print("Loading quantized model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="llama"  # Specify the model type (LLaMA in this case)
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError("Failed to load the quantized model.")

    def organize_content(self, input_text, max_tokens=1024):
        """
        Organizes the given lecture transcript into a detailed and easy-to-understand explanation.

        :param input_text: The raw lecture transcript.
        :param max_tokens: Maximum number of tokens for the generated output.
        :return: A detailed explanation of the lecture content.
        """
        # Improved prompt for detailed explanation, not just summary
        prompt = f"""
        You are a helpful teaching assistant. The following is a transcript of a lecture that a student missed.
        Please generate a **detailed explanation** of the lecture in a **step-by-step format**, breaking down complex topics into simpler terms.
        Include **examples** where appropriate to clarify each concept, and provide **in-depth explanations** for any technical terms.
        
        Lecture Transcript:
        {input_text}

        Detailed Explanation (with examples, explanations, and clarifications):
        """

        try:
            # Generate detailed content
            print("Generating detailed explanation...")
            response = self.model(
                prompt,
                max_new_tokens=max_tokens,  # Increase token limit for detailed explanation
                temperature=0.7,
                top_p=0.95
            )
            print("Detailed explanation generation completed.")
            clean_summary = self.clean_output(response)
            return clean_summary
        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    def clean_output(self, output_text):
        """
        Cleans and formats the generated output to remove unwanted repetitions and extra text.

        :param output_text: The raw text generated by the model.
        :return: Cleaned and formatted detailed explanation.
        """
        # Example of cleaning redundant repetitions or unwanted filler
        output_text = output_text.replace("students can suggest resources for additional study materials for additional resources", "Suggested resources for further study include:")
        
        # Further cleaning steps (you can customize this as needed)
        cleaned_text = " ".join(output_text.split())  # Remove extra spaces, line breaks, etc.
        
        # More specific cleaning rules can be applied depending on the output structure
        return cleaned_text