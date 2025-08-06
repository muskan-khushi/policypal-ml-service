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

# --- Data Models (Unchanged) ---
class FinalResponse(BaseModel):
    decision: str
    amount_covered: Optional[str] 
    justification: List[str]
    narrative_response: str

class ExtractedQuotes(BaseModel):
    quotes: List[str]

class DecisionResponse(BaseModel):
    decision: Optional[str] = "Could Not Determine"
    amount_covered: Optional[str] = "Not Specified" 

# --- HELPER FUNCTION TO CLEAN LLM OUTPUT (Unchanged) ---
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

            # --- ENHANCED PROMPT #1: Query Expansion ---
            expansion_prompt_text = f"""You are an expert insurance policy analyst with 15+ years of experience. Your task is to transform a user's simple question into a comprehensive search query that will effectively retrieve relevant policy clauses from a vector database.

ANALYSIS APPROACH:
- Consider coverage scope, exclusions, limits, conditions, and eligibility requirements
- Think about related insurance terminology and synonyms
- Include both direct coverage and potential exclusions
- Focus on actionable policy language

USER'S ORIGINAL QUERY: "{query}"

Create an enhanced search query that includes:
1. The core concept from the user's question
2. Related insurance terms and synonyms  
3. Potential policy sections that might contain relevant information
4. Both positive coverage terms and exclusionary language

ENHANCED SEARCH QUERY:"""
            
            expanded_search_query = self.llm.invoke(expansion_prompt_text).content
            
            retrieved_docs = retriever.invoke(expanded_search_query)
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            
            if not retrieved_docs or not context.strip():
                return FinalResponse(decision="Could Not Determine", amount_covered="Not Specified", justification=[], narrative_response=f"Could not find information regarding '{query}'.")
            
            # --- ENHANCED PROMPT #2: Quote Extraction ---
            extraction_prompt_text = f'''You are a legal document analyst specializing in insurance policy interpretation. Your expertise is in identifying the most relevant and actionable policy language.

TASK: Extract sentences from the provided context that directly address the user's query. Focus on:
- Specific coverage statements (what IS covered)
- Explicit exclusions (what is NOT covered) 
- Monetary limits, deductibles, or percentages
- Eligibility conditions and requirements
- Claims procedures or authorization requirements

QUALITY CRITERIA:
- Each quote must be a complete, grammatically correct sentence
- Prioritize specific, actionable language over general statements
- Include both favorable and unfavorable provisions
- Ensure quotes provide clear, definitive information

CONTEXT:
---
{context}
---

USER'S QUERY: {query}

Extract the most relevant sentences that a claims adjudicator would need to make a coverage determination. Return ONLY a valid JSON object with the key "quotes" containing an array of relevant sentence quotes.

JSON Response:'''
            quotes_response_str = self.llm.invoke(extraction_prompt_text).content
            
            cleaned_quotes_json = extract_json_from_string(quotes_response_str)
            if not cleaned_quotes_json:
                 return FinalResponse(decision="Error", amount_covered="N/A", justification=[], narrative_response="Could not parse justification quotes from AI response.")
            extracted_justifications = ExtractedQuotes.parse_raw(cleaned_quotes_json).quotes
            
            if not extracted_justifications:
                return FinalResponse(decision="Could Not Determine", amount_covered="Not Specified", justification=[], narrative_response=f"Found general information but no specific clauses for '{query}'.")

            quotes_for_decision = "\n".join(extracted_justifications)
            
            # --- ENHANCED PROMPT #3: Decision Making ---
            decision_prompt_text = f'''You are a senior insurance claims adjudicator with 20+ years of experience in policy interpretation and claims decisions. You must make a coverage determination based strictly on the provided policy excerpts.

DECISION FRAMEWORK:
1. COVERAGE ANALYSIS: Is there explicit language covering this situation?
2. EXCLUSION ANALYSIS: Are there any exclusions that would deny coverage?
3. CONDITIONS ANALYSIS: Are all policy conditions and requirements met?
4. LIMITS ANALYSIS: What monetary amounts, percentages, or limits apply?

DECISION CATEGORIES:
- "Approved": Clear, unambiguous coverage with no applicable exclusions
- "Rejected": Explicit exclusion exists or coverage is clearly not provided
- "Partial Approval": Limited coverage applies or conditions must be met
- "Further Information Required": Policy provisions exist but additional details needed
- "Could Not Determine": Policy language is ambiguous or insufficient

POLICY EXCERPTS:
---
{quotes_for_decision}
---

USER'S COVERAGE QUESTION: {query}

Based solely on these policy excerpts, make your determination. If specific dollar amounts, percentages, or limits are mentioned, include them exactly as stated.

Respond with a valid JSON object containing:
- "decision": Your coverage determination (use exact categories above)
- "amount_covered": Specific amount/limit from policy, or "Not Specified" if none mentioned

JSON Response:'''
            decision_response_str = self.llm.invoke(decision_prompt_text).content

            cleaned_decision_json = extract_json_from_string(decision_response_str)
            if not cleaned_decision_json:
                return FinalResponse(decision="Error", amount_covered="N/A", justification=[], narrative_response="Could not parse final decision from AI response.")
            json_decision = DecisionResponse.parse_raw(cleaned_decision_json)

            final_data_for_narrative = {"decision": json_decision.decision, "amount_covered": json_decision.amount_covered, "justification_quotes": extracted_justifications}
            
            # --- ENHANCED PROMPT #4: Narrative Generation ---
            narrative_prompt_text = f'''You are "PolicyPal," a trusted AI insurance advisor known for clear, empathetic, and helpful communication. Your role is to translate complex policy decisions into plain English that customers can easily understand and act upon.

COMMUNICATION STYLE:
- Professional yet approachable and empathetic
- Use "your policy" language to make it personal  
- Explain insurance jargon in simple terms
- Be direct and honest, but supportive
- Provide actionable next steps when appropriate
- Write in plain text WITHOUT any markdown formatting (no **, *, #, etc.)
- Use clear paragraph breaks and natural language emphasis

RESPONSE STRUCTURE:
1. START with a clear, direct answer to their question
2. EXPLAIN the reasoning using specific policy language (reference key phrases, don't reproduce entire quotes)
3. CLARIFY any important conditions, limits, or requirements
4. END with helpful next steps or recommendations

FORMATTING REQUIREMENTS:
- Use plain text only - no bold, italic, or markdown formatting
- Use natural language for emphasis (e.g., "importantly" instead of **Important**)
- Separate sections with line breaks, not headers
- Write in a conversational, flowing style

DECISION DATA:
{json.dumps(final_data_for_narrative, indent=2)}

USER'S ORIGINAL QUESTION: "{query}"

As PolicyPal, write a comprehensive yet accessible response that helps the user understand their coverage situation. If the decision is uncertain, acknowledge this honestly and guide them on next steps. Remember: use only plain text formatting.

Your response:'''
            narrative_text = self.llm.invoke(narrative_prompt_text).content
            
            return FinalResponse(
                decision=json_decision.decision,
                amount_covered=str(json_decision.amount_covered),
                justification=extracted_justifications,
                narrative_response=narrative_text.strip()
            )
        finally:
            os.remove(temp_file_path)

rag_processor = RAGProcessor()

