import os
import json
import tempfile
import re
from io import BytesIO
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional, List
from pydantic.v1 import BaseModel, Field
from dotenv import load_dotenv

# This line loads your .env file for local testing and deployment
load_dotenv()

# --- Data Models ---
class FinalResponse(BaseModel):
    decision: str
    # --- FIX: Changed from float to str to be more flexible ---
    amount_covered: Optional[str] 
    justification: List[str]
    narrative_response: str

class ExtractedQuotes(BaseModel):
    quotes: List[str]

class DecisionResponse(BaseModel):
    decision: Optional[str] = "Could Not Determine"
    # --- FIX: Changed from float to str to handle non-numeric LLM responses ---
    amount_covered: Optional[str] = "Not Specified" 

# --- HELPER FUNCTION TO CLEAN LLM OUTPUT ---
def extract_json_from_string(text: str) -> Optional[str]:
    """
    Finds and extracts the first valid JSON object from a string.
    Handles cases where the LLM adds conversational text around the JSON.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

class RAGProcessor:
    def __init__(self):
        load_dotenv()
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama3-8b-8192",
            groq_api_key=os.environ.get("GROQ_API_KEY")
        )
        self.embedding_model = CohereEmbeddings(
            cohere_api_key=os.environ.get("COHERE_API_KEY"),
            model="embed-english-light-v3.0" 
        )
        print(">> Fully Cloud RAG Processor Ready.")
    
    def process_document_and_query(self, file_bytes: bytes, query: str) -> FinalResponse:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            
            if not documents:
                return FinalResponse(decision="Error", amount_covered="N/A", justification=[], narrative_response="Could not read the PDF.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
            chunks = text_splitter.split_documents(documents)
            vector_store = Chroma.from_documents(chunks, self.embedding_model)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})

            expansion_prompt_text = f"Rewrite the user's query into a detailed question for an insurance policy search. Query: '{query}'"
            expanded_search_query = self.llm.invoke(expansion_prompt_text).content
            
            retrieved_docs = retriever.invoke(expanded_search_query)
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            
            if not retrieved_docs or not context.strip():
                return FinalResponse(decision="Could Not Determine", amount_covered="Not Specified", justification=[], narrative_response=f"Could not find information regarding '{query}'.")
            
            extraction_prompt_text = f'From the CONTEXT below, extract a list of all sentences relevant to the USER\'S QUERY. Your response MUST be a valid JSON object with a single key "quotes", which is a list of strings.\n\nCONTEXT:\n---\n{context}\n---\nUSER\'S QUERY: {query}\n---\nJSON Response:'
            quotes_response_str = self.llm.invoke(extraction_prompt_text).content
            
            cleaned_quotes_json = extract_json_from_string(quotes_response_str)
            if not cleaned_quotes_json:
                 return FinalResponse(decision="Error", amount_covered="N/A", justification=[], narrative_response="Could not parse justification quotes from AI response.")
            extracted_justifications = ExtractedQuotes.parse_raw(cleaned_quotes_json).quotes
            
            if not extracted_justifications:
                return FinalResponse(decision="Could Not Determine", amount_covered="Not Specified", justification=[], narrative_response=f"Found general information but no specific clauses for '{query}'.")

            quotes_for_decision = "\n".join(extracted_justifications)
            decision_prompt_text = f'Based ONLY on the following policy QUOTES, make a final decision (e.g., "Approved", "Rejected"). Your response MUST be a valid JSON object with the keys "decision" and "amount_covered" (which can be a number or a descriptive string like "Not Specified").\n\nQUOTES:\n---\n{quotes_for_decision}\n---\nUSER\'S QUERY: {query}\n---\nJSON Response:'
            decision_response_str = self.llm.invoke(decision_prompt_text).content

            cleaned_decision_json = extract_json_from_string(decision_response_str)
            if not cleaned_decision_json:
                return FinalResponse(decision="Error", amount_covered="N/A", justification=[], narrative_response="Could not parse final decision from AI response.")
            json_decision = DecisionResponse.parse_raw(cleaned_decision_json)

            final_data_for_narrative = {"decision": json_decision.decision, "amount_covered": json_decision.amount_covered, "justification_quotes": extracted_justifications}
            narrative_prompt_text = f"You are a friendly support AI. Convert the following data into a gentle, easy-to-understand paragraph.\n\nDATA:\n{json.dumps(final_data_for_narrative, indent=2)}\n---\nYour friendly paragraph:"
            narrative_text = self.llm.invoke(narrative_prompt_text).content
            
            return FinalResponse(
                decision=json_decision.decision,
                amount_covered=str(json_decision.amount_covered), # Ensure it's a string for the final response
                justification=extracted_justifications,
                narrative_response=narrative_text.strip()
            )
        finally:
            os.remove(temp_file_path)

rag_processor = RAGProcessor()

