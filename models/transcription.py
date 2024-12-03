from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import librosa

class TranscriptionPipeline:
    def __init__(self, model_id="openai/whisper-large-v3-turbo"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load the model and processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Create the transcription pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,  # Process 30-second chunks
            batch_size=8,       # Adjust based on your GPU memory
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe_audio(self, audio_path, language="english"):
        
        try:
            
            
        # Load audio using librosa
            audio, sr = librosa.load(audio_path, sr=16000)
        
        # Run the transcription pipeline
            result = self.pipe(audio, generate_kwargs={"language": language})
        
        # Extract transcription text and handle missing chunks
            transcription = result.get("text", "No transcription available.")
            chunks = result.get("chunks", [])  # Default to an empty list if chunks are missing
        
            return transcription, chunks
        except Exception as e:
            
            print(f"Error during transcription: {e}")
            return None, []

    def save_transcription(self, text, output_path="transcription.txt"):
        # Save the transcription to a file
        with open(output_path, "w") as f:
            f.write(text)

