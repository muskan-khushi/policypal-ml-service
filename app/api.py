from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from .core.logic import rag_processor, FinalResponse 

router = APIRouter()

@router.post("/process")
async def process_document_and_get_answer(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """Accepts a PDF document and a text query, returning a structured JSON answer."""
    
    # Add logging to see what's happening
    print(f"📄 Received file: {file.filename} ({file.content_type})")
    print(f"❓ Query: {query}")
    
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        print("🔄 Reading file bytes...")
        file_bytes = await file.read()
        print(f"✅ File read successfully. Size: {len(file_bytes)} bytes")
        
        print("🧠 Processing with RAG processor...")
        result = rag_processor.process_document_and_query(file_bytes=file_bytes, query=query)
        print("✅ RAG processing complete")
        
        print("📤 Converting result to dict...")
        response_data = result.dict()
        print("✅ Response ready to send")
        
        return response_data

    except Exception as e:
        import traceback
        print("---! PYTHON SERVER ERROR !---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        print("---------------------------")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
