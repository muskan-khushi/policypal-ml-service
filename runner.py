# doc_qa_backend/runner.py

import sys
import json
from app.core.logic import rag_processor

def run_analysis(file_path, query):
    """
    This function now passes the file_path directly to the RAG processor.
    """
    try:
        # Call the main processing function with the file_path
        result = rag_processor.process_document_and_query(file_path=file_path, query=query)
        
        # Print the final result as a JSON string
        print(json.dumps(result.dict()))

    except Exception as e:
        import traceback
        error_response = {
            "error": "Python script failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_response), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        error_msg = {"error": "Invalid arguments. Expected file_path and query."}
        print(json.dumps(error_msg), file=sys.stderr)
        sys.exit(1)
    
    file_path_arg = sys.argv[1]
    query_arg = sys.argv[2]
    
    run_analysis(file_path_arg, query_arg)