from flask import Flask, request, render_template, send_file
from models.transcription import TranscriptionPipeline
from models.summary import Summarizer
from models.rag_qna import RAGPipeline
import os

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'uploads'
TEXT_FOLDER = 'text'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEXT_FOLDER'] = TEXT_FOLDER

# Initialize pipelines
transcription_pipeline = TranscriptionPipeline()
summarizer = Summarizer()
rag_pipeline = RAGPipeline(persist_directory="doc_db")


@app.route('/')
def index():
    """
    Renders the main index page with options to upload and query.
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    if file.filename.endswith(".mp4") or file.filename.endswith(".wav"):
        # Mock transcription and summarization process
        transcribed_text = "This is a placeholder transcription for the uploaded file."
        # Replace this with your actual transcription logic:
        # transcribed_text, _ = transcription_pipeline.transcribe_audio(file_path)

        # Use the mocked summary for now
        summarized_content = {
            "Introduction": "This section introduces the main topics covered in the transcription.",
            "Key Points": "Here are the key takeaways and highlights from the content.",
            "Conclusion": "This section provides a final summary or practical recommendations."
        }

        return render_template('summary.html', summary=summarized_content)
    else:
        return "Unsupported file type", 400


@app.route('/download-summary')
def download_summary():
    """
    Provides a download link for the generated summary.
    """
    summary_file_path = os.path.join(app.config['TEXT_FOLDER'], "summary.txt")
    if os.path.exists(summary_file_path):
        return send_file(summary_file_path, as_attachment=True)
    else:
        return "No summary available", 404


@app.route('/upload-doc', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process the document for RAG
    documents = rag_pipeline.load_documents(app.config['UPLOAD_FOLDER'])
    text_chunks = rag_pipeline.process_documents(documents)
    rag_pipeline.create_vectorstore(text_chunks)

    # Initialize the QA chain after processing the vector store
    rag_pipeline.initialize_qa_chain(model_name="google/flan-t5-base")

    return "Document uploaded and processed successfully"


@app.route('/query', methods=['POST'])
def query():
    """
    Handles user queries to the RAG pipeline.
    """
    question = request.form.get('question')
    if not question:
        return "No query provided", 400

    try:
        result, sources = rag_pipeline.answer_query(question)
    except Exception as e:
        return f"Error processing query: {str(e)}", 500

    return render_template('results.html', question=question, result=result, sources=sources)


if __name__ == '__main__':
    app.run(debug=True)
