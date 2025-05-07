import asyncio
import json
import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx


from chatbot import FineTunedChatbot
from auth.middleware import verify_jwt_and_timestamp

# Configuration
RESPONSE_TIMEOUT = 60  # HTTP response timeout in seconds

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chatbot instance
chatbot = None

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code: initialize chatbot
    global chatbot
    logger.info("Initializing chatbot model...")
    chatbot = FineTunedChatbot()
    logger.info("Chatbot model initialized successfully")
    
    # Yield control back to FastAPI
    yield
    
    # Shutdown code: cleanup resources
    logger.info("Shutting down chatbot and releasing resources...")
    # If your chatbot needs specific cleanup, add it here
    # For example: await chatbot.close() 
    logger.info("Shutdown complete")

# Initialize the application with lifespan
app = FastAPI(
    title="Fine-Tuned Chatbot API",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # During development/testing, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)



class QuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None
    stream: bool = False
    stream_mode: Optional[str] = "chunk"  # "chunk" or "token"

@app.post("/ask",dependencies=[Depends(verify_jwt_and_timestamp)])
async def ask(request: QuestionRequest,):
    """
    Ask a question to the chatbot.
    - stream: If true, returns a streaming response
    - stream_mode: "chunk" for fake streaming (chunks of a complete response),
                  "token" for true token-by-token streaming
    """
    if not chatbot:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is still initializing. Please try again in a moment."}
        )
    
    if request.stream:
        # Choose streaming mode
        stream_mode = getattr(request, "stream_mode", "chunk")
        
        if stream_mode == "token":
            # True token-by-token streaming
            return StreamingResponse(
                generate_true_stream_response(request.question, request.context),
                media_type="text/event-stream"
            )
        else:
            # Chunk-based streaming (original method)
            return StreamingResponse(
                generate_stream_response(request.question, request.context),
                media_type="text/event-stream"
            )
    else:
        # Regular non-streaming response
        try:
            start_time = time.time()
            result = chatbot.ask(request.question, request.context)
            processing_time = time.time() - start_time
            
            return {
                "answer": result['answer'],
                "complete": result['complete'],
                "processing_time": processing_time
            }
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to process request: {str(e)}"}
            )

async def generate_stream_response(question, context=None):
    """
    Generator for streaming responses.
    """
    full_response = ""
    try:
        # Send initial processing message
        yield f"data: {json.dumps({'content': '', 'status': 'processing', 'done': False})}\n\n"
        
        # Add keepalive messages to prevent timeout
        keepalive_task = asyncio.create_task(send_keepalive_pings())
        
        # Process response in background thread to not block event loop
        start_time = time.time()
        result = await asyncio.to_thread(chatbot.ask, question, context)
        processing_time = time.time() - start_time
        
        # Cancel keepalive task
        keepalive_task.cancel()
        
        # Get the full response text
        full_response = result['answer']
        
        # Stream the response in chunks
        words = full_response.split()
        total_words = len(words)
        
        # Calculate chunk size and delay
        chunk_size = max(3, total_words // 10)  # At least 3 words per chunk, up to 10 chunks
        delay = min(0.3, processing_time / 20)  # Scale delay with processing time
        
        # Stream response in chunks
        for i in range(0, total_words, chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            data = {
                "content": chunk,
                "done": False,
                "status": "streaming"
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(delay)
        
        # Send completion message
        yield f"data: {json.dumps({
            'content': '',
            'done': True,
            'processing_time': processing_time,
            'status': 'complete'
        })}\n\n"
        
    except Exception as e:
        logger.error(f"Error in stream generation: {str(e)}")
        # Send error message
        yield f"data: {json.dumps({
            'error': str(e),
            'done': True,
            'status': 'error'
        })}\n\n"

async def generate_true_stream_response(question, context=None):
    """
    Generator for token streaming with word batching for efficiency.
    Sends 6-7 words at a time to reduce response count.
    """
    try:
        # Send initial processing message
        yield f"data: {json.dumps({'content': '', 'status': 'processing', 'done': False})}\n\n"
        
        # Track the full response and batched content
        full_response = ""
        batch_content = ""
        word_count = 0
        target_word_count = 6  # Number of words to batch together
        
        # Start timer for processing time tracking
        start_time = time.time()
        
        # Stream tokens from the model
        for token in chatbot.ask_stream(question, context):
            # Determine if this is a complete new response or incremental update
            if len(token) > len(full_response) + 20:
                # This is a complete new response after </think> tag
                # Send it immediately without batching
                data = {
                    "content": token,
                    "done": False,
                    "status": "streaming"
                }
                yield f"data: {json.dumps(data)}\n\n"
                
                # Update tracking variables
                full_response = token
                batch_content = ""
                word_count = 0
            else:
                # This is an incremental update
                # Get just the new content
                new_content = token[len(full_response):] if full_response else token
                
                if new_content.strip():
                    # Add to batch
                    batch_content += new_content
                    full_response = token
                    
                    # Count words in batch
                    current_words = len(batch_content.split())
                    
                    # Send batch when it reaches target size or contains sentence-ending punctuation
                    if (current_words >= target_word_count or 
                        any(mark in batch_content for mark in ['.', '!', '?', '\n'])):
                        
                        data = {
                            "content": batch_content,
                            "done": False,
                            "status": "streaming"
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        
                        # Reset batch
                        batch_content = ""
            
            # Small delay between processing tokens
            await asyncio.sleep(0.001)  # Minimal delay since we're batching
        
        # Send any remaining content in the final batch
        if batch_content.strip():
            data = {
                "content": batch_content,
                "done": False,
                "status": "streaming"
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Use chatbot's built-in method to check if response is complete
        is_complete = chatbot._is_response_complete(full_response)
        
        # Send completion message
        yield f"data: {json.dumps({
            'content': '',
            'done': True,
            'complete': is_complete,
            'processing_time': processing_time,
            'status': 'complete'
        })}\n\n"
        
    except Exception as e:
        logger.error(f"Error in stream generation: {str(e)}")
        # Send error message
        yield f"data: {json.dumps({
            'error': str(e),
            'done': True,
            'status': 'error'
        })}\n\n"

async def send_keepalive_pings():
    """Send empty comments as keepalive messages every 5 seconds"""
    try:
        while True:
            await asyncio.sleep(5)
            # This comment will be ignored by SSE clients but keeps the connection alive
            logger.debug("Sending keepalive ping")
    except asyncio.CancelledError:
        logger.debug("Keepalive task cancelled")
        pass

class ConversationRequest(BaseModel):
    message: str

async def stream_audio_from_tts(text_response):
    """
    Stream audio from TTS service for the given text response.
    
    Args:
        text_response: The text to convert to speech
        
    Yields:
        Audio data chunks
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            logger.info(f"Sending text to TTS service: {text_response[:30]}...")
            
            # Stream the response from the audio service
            async with client.stream(
                "POST", 
                "http://localhost:8080/generate-audio",
                json={"message": text_response},
                timeout=60.0
            ) as response:
                if response.status_code != 200:
                    error_text = await response.text()
                    logger.error(f"TTS service error: {response.status_code}, {error_text}")
                    yield b''  # Empty byte to avoid breaking the stream
                    return
                        
                # Stream audio bytes
                async for chunk in response.aiter_bytes():
                    yield chunk
                    
    except Exception as e:
        logger.error(f"Error streaming audio from TTS service: {str(e)}")
        yield b''  # Empty byte to avoid breaking the stream

@app.post("/chat/text")
async def chat_text(request: ConversationRequest):
    """
    Have a casual conversation with the chatbot.
    Returns a friendly, brief conversational response as JSON.
    """
    if not chatbot:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is still initializing. Please try again in a moment."}
        )
    
    try:
        # Generate text response from chatbot
        start_time = time.time()
        result = chatbot.get_conversational_response(request.message)
        text_processing_time = time.time() - start_time
        
        # Return JSON response with text and metadata
        return {
            "text_response": result['answer'],
            "complete": result['complete'],
            "processing_time": text_processing_time
        }
        
    except Exception as e:
        logger.error(f"Error processing text request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process conversation: {str(e)}"}
        )

class ConversationExampleRequest(BaseModel):
    content: str

@app.post('/chat/example')
async def chat_example_text(request: ConversationExampleRequest):
    """
    Generate examples related to the provided content.
    Returns examples that help illustrate the content.
    """
    if not chatbot:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is still initializing. Please try again in a moment."}
        )
    
    try:
        # Generate examples based on the provided content
        start_time = time.time()
        
        # Check for empty content
        if not request.content or len(request.content.strip()) < 10:
            return JSONResponse(
                status_code=400,
                content={"error": "Please provide sufficient content for example generation"}
            )
        
        # Use the specialized example prompt method from the chatbot
        prompt = chatbot._generate_examples_prompt(request.content, example_count=2)
        
        # Generate examples using the prompt
        inputs = chatbot.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(chatbot.model.device) for key, value in inputs.items()}
        
        # Generate response with appropriate parameters for examples
        outputs = chatbot.model.generate(
            **inputs,
            max_new_tokens=200,      # Examples should be concise
            do_sample=True,
            temperature=0.7,         # Slightly higher creativity for examples
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=chatbot.tokenizer.eos_token_id
        )
        
        response = chatbot.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        examples_text = chatbot._get_reply(response)
        processing_time = time.time() - start_time
        
        # Return JSON response with examples and metadata
        return {
            "examples": examples_text,
            "complete": chatbot._is_response_complete(response),
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error generating examples: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate examples: {str(e)}"}
        )

@app.get('/chat/question')
async def get_question_prompt():
        """
        Provides a random question prompt to help users get started.
        Returns a question prompt as JSON or audio.
        """
        if not chatbot:
            return JSONResponse(
            status_code=503,
            content={"error": "Model is still initializing. Please try again in a moment."}
            )
        
        try:
            
            # Instead of generating audio here, redirect to the TTS service
            tts_url = "http://localhost:8080/get-question-prompt"
            
            async with httpx.AsyncClient() as client:
                tts_response = await client.get(
                tts_url,
                headers={"Accept": "audio/mpeg"}
                )
                
                if tts_response.status_code != 200:
                    logger.error(f"TTS service error: {tts_response.status_code}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": "Failed to get audio from TTS service"}
                    )
                    
                    # Return the audio content from the TTS service
                    # Return the streaming audio response
                return StreamingResponse(
                    content=tts_response.aiter_bytes(),
                    media_type="audio/mpeg",
                    headers={
                        "Content-Disposition": "attachment; filename=question_prompt.mp3"
                    }
                )
                
        except Exception as e:
            logger.error(f"Error generating question prompt: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to generate question prompt: {str(e)}"}
            )

@app.post("/chat/audio")
async def chat_audio(request: ConversationRequest):
    """
    Generate audio for the provided text message.
    Returns streaming audio data.
    """
    try:
        # Return streaming audio response
        return StreamingResponse(
            stream_audio_from_tts(request.message),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=response.mp3"
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing audio request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate audio: {str(e)}"}
        )

class ConversationQuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None

@app.post("/chat/ask")
def ask_question(request: ConversationQuestionRequest):
    """
    Ask a question to the chatbot and get a response.
    Returns a JSON response with the answer.
    """
    if not chatbot:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is still initializing. Please try again in a moment."}
        )
    
    try:
        # Generate text response from chatbot
        start_time = time.time()
        result = chatbot.get_conversational_response(request.question, request.context)
        processing_time = time.time() - start_time
        
        # Return JSON response with text and metadata
        return {
            "answer": result['answer'],
            "complete": result['complete'],
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error processing question request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process question: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": chatbot is not None
    }

# Add these new request models to app.py
class PresentationRequest(BaseModel):
    topic: str
    context: Optional[str] = None
    stream: bool = False

class BulletPointsRequest(BaseModel):
    topic: str
    context: Optional[str] = None
    stream: bool = False



@app.post("/bullet-points",dependencies=[Depends(verify_jwt_and_timestamp)])
async def create_bullet_points(request: BulletPointsRequest):
    """
    Create bullet points on the given topic.
    - stream: If true, returns a streaming response
    """
    if not chatbot:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is still initializing. Please try again in a moment."}
        )
    
    if request.stream:
        # Streaming response
        return StreamingResponse(
            generate_bullet_points_stream(request.topic, request.context),
            media_type="text/event-stream"
        )
    else:
        # Regular non-streaming response
        try:
            start_time = time.time()
            result = chatbot.create_bullet_points(request.topic, request.context, stream=False)
            processing_time = time.time() - start_time
            
            return {
                "bullet_points": result['answer'],
                "complete": result['complete'],
                "processing_time": processing_time
            }
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to process request: {str(e)}"}
            )

# Add these streaming generator functions
@app.post("/presentation",dependencies=[Depends(verify_jwt_and_timestamp)])
async def create_presentation(request: PresentationRequest):
    """
    Create a presentation on the given topic.
    - stream: If true, returns a streaming response (not recommended)
    """
    if not chatbot:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is still initializing. Please try again in a moment."}
        )
    
    # Note: According to chatbot.py, streaming isn't implemented for presentations anymore
    if request.stream:
        return JSONResponse(
            status_code=400,
            content={"error": "Streaming is not supported for presentations. Please set stream=false."}
        )
    else:
        # Regular non-streaming response
        try:
            start_time = time.time()
            result = chatbot.create_presentation(request.topic, request.context)
            processing_time = time.time() - start_time
            
            return {
                "presentation": result['answer'],
                "complete": result['complete'],
                "processing_time": processing_time
            }
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to process request: {str(e)}"}
            )

async def generate_bullet_points_stream(topic, context=None):
    """
    Generator for streaming bullet points content.
    """
    try:
        # Send initial processing message
        yield f"data: {json.dumps({'content': '', 'status': 'processing', 'done': False})}\n\n"
        
        # Track the full response and batched content
        full_response = ""
        batch_content = ""
        word_count = 0
        target_word_count = 8  # Slightly larger batches for bullet points
        
        # Start timer for processing time tracking
        start_time = time.time()
        
        # Stream tokens from the model
        for token in chatbot.create_bullet_points(topic, context, stream=True):
            # Determine if this is a complete new response or incremental update
            if len(token) > len(full_response) + 20:
                # This is a complete new response after </think> tag
                # Send it immediately without batching
                data = {
                    "content": token,
                    "done": False,
                    "status": "streaming"
                }
                yield f"data: {json.dumps(data)}\n\n"
                
                # Update tracking variables
                full_response = token
                batch_content = ""
                word_count = 0
            else:
                # This is an incremental update
                # Get just the new content
                new_content = token[len(full_response):] if full_response else token
                
                if new_content.strip():
                    # Add to batch
                    batch_content += new_content
                    full_response = token
                    
                    # Count words in batch
                    current_words = len(batch_content.split())
                    
                    # Send batch when it reaches target size or contains bullet point markers
                    if (current_words >= target_word_count or 
                        any(mark in batch_content for mark in ['.', '!', '?', '\n', '*', '-'])):
                        
                        data = {
                            "content": batch_content,
                            "done": False,
                            "status": "streaming"
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        
                        # Reset batch
                        batch_content = ""
            
            # Small delay between processing tokens
            await asyncio.sleep(0.001)
        
        # Send any remaining content in the final batch
        if batch_content.strip():
            data = {
                "content": batch_content,
                "done": False,
                "status": "streaming"
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Use chatbot's built-in method to check if response is complete
        is_complete = chatbot._is_response_complete(full_response)
        
        # Send completion message
        yield f"data: {json.dumps({
            'content': '',
            'done': True,
            'complete': is_complete,
            'processing_time': processing_time,
            'status': 'complete'
        })}\n\n"
        
    except Exception as e:
        logger.error(f"Error in stream generation: {str(e)}")
        # Send error message
        yield f"data: {json.dumps({
            'error': str(e),
            'done': True,
            'status': 'error'
        })}\n\n"


if __name__ == "__main__":
    # Configure uvicorn with extended timeouts
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        timeout_keep_alive=RESPONSE_TIMEOUT,
        workers=1,
        reload=True
    )