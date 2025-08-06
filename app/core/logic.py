import os
import json
from io import BytesIO
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional, List
from pydantic.v1 import BaseModel, Field
from dotenv import load_dotenv

# This line loads the GROQ_API_KEY from your .env file
load_dotenv()

# --- Data Models (These are the same as before) ---
class FinalResponse(BaseModel):
    decision: str
    amount_covered: Optional[float]
    justification: List[str]
    narrative_response: str

class ExtractedQuotes(BaseModel):
    quotes: List[str]

class DecisionResponse(BaseModel):
    decision: Optional[str] = "Could Not Determine"
    amount_covered: Optional[float] = 0.0

class RAGProcessor:
    def __init__(self):
        # --- THIS IS THE UPGRADE ---
        # We are now using the ultra-fast Groq cloud service with Llama 3
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama3-8b-8192",
            groq_api_key=os.environ.get("GROQ_API_KEY")
        )
        # --- The rest of the setup is the same ---
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        print(">> Cloud RAG Processor Ready.")
    
    def process_document_and_query(self, file_bytes: bytes, query: str) -> FinalResponse:
        # This entire function's logic remains the same as your last working version.
        # It will now just use the much faster and more powerful Groq LLM.
        loader = PyPDFLoader(BytesIO(file_bytes))
        documents = loader.load()
        if not documents:
            return FinalResponse(decision="Error", amount_covered=0, justification=[], narrative_response="Could not read the PDF.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(chunks, self.embedding_model)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        expansion_prompt_text = f"Rewrite the user's query into a detailed question for an insurance policy search. Query: '{query}'"
        expanded_search_query = self.llm.invoke(expansion_prompt_text).content
        
        retrieved_docs = retriever.invoke(expanded_search_query)
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        if not retrieved_docs or not context.strip():
            return FinalResponse(decision="Could Not Determine", amount_covered=0, justification=[], narrative_response=f"Could not find information regarding '{query}'.")
        
        extraction_prompt_text = f'From the CONTEXT below, extract a list of all sentences relevant to the USER\'S QUERY. Your response MUST be a valid JSON object with a single key "quotes", which is a list of strings.\n\nCONTEXT:\n---\n{context}\n---\nUSER\'S QUERY: {query}\n---\nJSON Response:'
        quotes_response_str = self.llm.invoke(extraction_prompt_text).content
        extracted_justifications = ExtractedQuotes.parse_raw(quotes_response_str).quotes
        
        if not extracted_justifications:
             return FinalResponse(decision="Could Not Determine", amount_covered=0, justification=[], narrative_response=f"Found general information but no specific clauses for '{query}'.")

        quotes_for_decision = "\n".join(extracted_justifications)
        decision_prompt_text = f'Based ONLY on the following policy QUOTES, make a final decision (e.g., "Approved", "Rejected"). Your response MUST be a valid JSON object with the keys "decision" and "amount_covered".\n\nQUOTES:\n---\n{quotes_for_decision}\n---\nUSER\'S QUERY: {query}\n---\nJSON Response:'
        decision_response_str = self.llm.invoke(decision_prompt_text).content
        json_decision = DecisionResponse.parse_raw(decision_response_str)

        final_data_for_narrative = {"decision": json_decision.decision, "justification_quotes": extracted_justifications}
        narrative_prompt_text = f"You are a friendly support AI. Convert the following data into a gentle, easy-to-understand paragraph.\n\nDATA:\n{json.dumps(final_data_for_narrative, indent=2)}\n---\nYour friendly paragraph:"
        narrative_text = self.llm.invoke(narrative_prompt_text).content
        
        return FinalResponse(
            decision=json_decision.decision,
            amount_covered=json_decision.amount_covered,
            justification=extracted_justifications,
            narrative_response=narrative_text.strip()
        )

rag_processor = RAGProcessor()