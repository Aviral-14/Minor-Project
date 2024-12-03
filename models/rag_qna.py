from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


class RAGPipeline:
    def __init__(self, persist_directory="doc_db"):
        # Initialize components
        self.persist_directory = persist_directory
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def load_documents(self, directory):
        loader = DirectoryLoader(
            directory, glob="*.pdf", loader_cls=UnstructuredPDFLoader
        )
        return loader.load()

    def process_documents(self, documents):
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        return text_splitter.split_documents(documents)

    def create_vectorstore(self, text_chunks):
        self.vectorstore = Chroma.from_documents(
            documents=text_chunks,
            embedding=self.embedding,
            persist_directory=self.persist_directory,
        )
        self.retriever = self.vectorstore.as_retriever()

    def initialize_qa_chain(self, model_name="google/flan-t5-base"):
        # Load the HuggingFace model and tokenizer
        hf_pipeline = pipeline("text2text-generation", model=model_name)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
        # Initialize the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=self.retriever, return_source_documents=True
        )

    def answer_query(self, query):
        if not self.qa_chain:
            raise ValueError("QA chain has not been initialized.")
        response = self.qa_chain.invoke({"query": query})
        return response["result"], response.get("source_documents", [])
