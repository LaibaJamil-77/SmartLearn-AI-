import os
import shutil
import jwt
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import mysql.connector
from mysql.connector import Error as MySQLError
import logging
from chatbot import (
    extract_text_from_file,
    split_text_into_chunks,
    generate_embeddings,
    create_faiss_index,
    retrieve_relevant_chunks,
    generate_extra_explanation,
    generate_slides_from_topic,
    generate_slides_from_document,
    create_ppt,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()

# Mount static files

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGroq(model="llama3-70b-8192", temperature=0)

DB_CONFIG = {
    "host": "mysql-3f774157-uzmameer814-cf8c.j.aivencloud.com",
    "user": "avnadmin",
    "password": "AVNS_l_UCKqIH6COH1XG128S",
    "database": "defaultdb",
    "port": 11340,
}

SECRET_KEY = "123456"  # TODO: Replace with a secure key in .env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    first_name: str | None = None
    last_name: str | None = None
    avatar_index: int | None = None

class UserUpdate(BaseModel):
    email: str | None = None
    password: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    avatar_index: int | None = None

class TopicRequest(BaseModel):
    topic: str
    instructions: str | None = None
    num_slides: int = 10
    lang_choice: str = "english"

class DocumentRequest(BaseModel):
    instructions: str | None = None
    num_slides: int = 10
    lang_choice: str = "english"

class QARequest(BaseModel):
    question: str
    context_type: str
    lang_choice: str = "english"
    context: str | None = None

class QuizRequest(BaseModel):
    slide_filename: str
    lang_choice: str = "english"

class SlideResponse(BaseModel):
    slides: list[str]
    explanations: dict[str, str]
    ppt_filename: str

class QuizResponse(BaseModel):
    quiz_content: str
    quiz_filename: str

class QAResponse(BaseModel):
    answer: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserHistoryResponse(BaseModel):
    topics: list[dict]
    qa_history: list[dict]

UPLOAD_DIR = "uploads"
PPT_DIR = "ppt_files"
QUIZ_DIR = "quiz_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PPT_DIR, exist_ok=True)
os.makedirs(QUIZ_DIR, exist_ok=True)

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        logger.info("Database connection established")
        return connection
    except MySQLError as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

def verify_password(plain_password, hashed_password):
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid password hash format")

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def check_slide_content_schema():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("DESCRIBE slide_content")
        columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        connection.close()
        if "explanation" not in columns:
            logger.error("Missing 'explanation' column in slide_content table")
            raise HTTPException(
                status_code=500,
                detail="Database schema error: 'explanation' column missing in slide_content table. Run: ALTER TABLE slide_content ADD COLUMN explanation TEXT;"
            )
        logger.info("slide_content schema verified, 'explanation' column exists")
    except MySQLError as e:
        logger.error(f"Error checking slide_content schema: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking slide_content schema: {str(e)}")

def generate_quiz(slides, context_type, lang_choice='english'):
    """Generate a quiz with 15 MCQs based on slide content."""
    slides_text = "\n".join([slide.strip() for slide in slides if slide.strip()])
    
    system_prompt = f"""You are an expert teacher tasked with creating a quiz with 15 multiple-choice questions (MCQs) based on the provided content. 
    - Each question should have 4 answer options, with only one correct answer.
    - Questions should cover key points from the content and vary in difficulty (easy, medium, hard).
    - Format the quiz in Markdown, with each question starting with '### Question [number]', followed by the question, options, and the correct answer marked clearly.
    - Respond in {'Roman Urdu' if lang_choice == 'roman_urdu' else 'English'}, using clear, student-friendly language.
    - Ensure questions are relevant to the {'topic' if context_type == 'topic' else 'document'} content.
    Content: {{slides_text}}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
    ])
    chain = prompt | llm
    quiz_content = chain.invoke({"slides_text": slides_text}).content
    
    # Save quiz to a Markdown file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    quiz_filename = f"quiz_{context_type}_{timestamp}.md"
    quiz_path = os.path.join(QUIZ_DIR, quiz_filename)
    with open(quiz_path, 'w', encoding='utf-8') as f:
        f.write(f"# Quiz: {context_type.capitalize()}\n\n")
        f.write(quiz_content)
    
    logger.info(f"Quiz saved as '{quiz_path}'.")
    return quiz_content, quiz_filename

@app.post("/register", response_model=dict)
async def register_user(user: UserRegister):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        query = "SELECT username FROM users WHERE username = %s OR email = %s"
        cursor.execute(query, (user.username, user.email))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username or email already exists")
        hashed_password = get_password_hash(user.password)
        query = "INSERT INTO users (username, email, password, first_name, last_name, avatar_index) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(query, (user.username, user.email, hashed_password, user.first_name, user.last_name, user.avatar_index))
        connection.commit()
        logger.info(f"User registered: {user.username}")
        return {"message": "User registered successfully"}
    except MySQLError as e:
        if e.errno == 1062:
            raise HTTPException(status_code=400, detail="Username or email already exists")
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM users WHERE username = %s"
        cursor.execute(query, (form_data.username,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="Incorrect username or password")
        if not verify_password(form_data.password, user["password"]):
            raise HTTPException(status_code=401, detail="Incorrect username or password")
        access_token = create_access_token(data={"sub": user["username"]})
        logger.info(f"User logged in: {form_data.username}")
        return {"access_token": access_token, "token_type": "bearer"}
    except MySQLError as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.get("/user", response_model=dict, dependencies=[Depends(get_current_user)])
async def get_user(current_user: str = Depends(get_current_user)):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query = "SELECT username, email, first_name, last_name, avatar_index FROM users WHERE username = %s"
        cursor.execute(query, (current_user,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        logger.info(f"Retrieved profile for user: {current_user}")
        return user
    except MySQLError as e:
        logger.error(f"Error fetching user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching user: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.put("/user", response_model=dict, dependencies=[Depends(get_current_user)])
async def update_user(user: UserUpdate, current_user: str = Depends(get_current_user)):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        updates = []
        values = []
        if user.email:
            updates.append("email = %s")
            values.append(user.email)
        if user.password:
            updates.append("password = %s")
            values.append(get_password_hash(user.password))
        if user.first_name:
            updates.append("first_name = %s")
            values.append(user.first_name)
        if user.last_name:
            updates.append("last_name = %s")
            values.append(user.last_name)
        if user.avatar_index is not None:
            updates.append("avatar_index = %s")
            values.append(user.avatar_index)
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        query = f"UPDATE users SET {', '.join(updates)} WHERE username = %s"
        values.append(current_user)
        cursor.execute(query, values)
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        connection.commit()
        logger.info(f"User updated: {current_user}")
        return {"message": "User updated successfully"}
    except MySQLError as e:
        logger.error(f"Error updating user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating user: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.get("/slides", response_model=list[dict], dependencies=[Depends(get_current_user)])
async def list_slides(current_user: str = Depends(get_current_user)):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT ppt_filename, slide_count, created_at, quiz_filename
            FROM slides
            WHERE user_id = (SELECT id FROM users WHERE username = %s)
        """
        cursor.execute(query, (current_user,))
        slides = cursor.fetchall()
        logger.info(f"Retrieved {len(slides)} slides for user: {current_user}")
        return slides
    except MySQLError as e:
        logger.error(f"Error listing slides: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing slides: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.delete("/slides/{filename}", response_model=dict, dependencies=[Depends(get_current_user)])
async def delete_slides(filename: str, current_user: str = Depends(get_current_user)):
    file_path = os.path.join(PPT_DIR, filename)
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        query = "SELECT quiz_filename FROM slides WHERE ppt_filename = %s AND user_id = (SELECT id FROM users WHERE username = %s)"
        cursor.execute(query, (filename, current_user))
        result = cursor.fetchone()
        quiz_filename = result[0] if result else None
        query = "DELETE FROM slides WHERE ppt_filename = %s AND user_id = (SELECT id FROM users WHERE username = %s)"
        cursor.execute(query, (filename, current_user))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Slide not found or not owned by user")
        connection.commit()
        if os.path.exists(file_path):
            os.remove(file_path)
        if quiz_filename and os.path.exists(os.path.join(QUIZ_DIR, quiz_filename)):
            os.remove(os.path.join(QUIZ_DIR, quiz_filename))
        logger.info(f"Deleted slide: {filename} and quiz: {quiz_filename} for user: {current_user}")
        return {"message": "Slide and associated quiz deleted successfully"}
    except MySQLError as e:
        logger.error(f"Error deleting slide: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting slide: {str(e)}")
    except OSError as e:
        logger.error(f"Error removing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error removing file: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.get("/qa-history", response_model=list[dict], dependencies=[Depends(get_current_user)])
async def get_qa_history(current_user: str = Depends(get_current_user)):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query = """
            SELECT question, answer, context_type, response_language, asked_at
            FROM qa_history
            WHERE user_id = (SELECT id FROM users WHERE username = %s)
        """
        cursor.execute(query, (current_user,))
        history = cursor.fetchall()
        logger.info(f"Retrieved {len(history)} QA history entries for user: {current_user}")
        return history
    except MySQLError as e:
        logger.error(f"Error fetching QA history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching QA history: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.get("/contexts", response_model=dict, dependencies=[Depends(get_current_user)])
async def list_contexts(current_user: str = Depends(get_current_user)):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query_topics = "SELECT topic_name FROM topics WHERE user_id = (SELECT id FROM users WHERE username = %s)"
        cursor.execute(query_topics, (current_user,))
        topics = [row["topic_name"] for row in cursor.fetchall()]
        query_docs = "SELECT filename FROM documents WHERE user_id = (SELECT id FROM users WHERE username = %s)"
        cursor.execute(query_docs, (current_user,))
        documents = [row["filename"] for row in cursor.fetchall()]
        logger.info(f"Retrieved {len(topics)} topics and {len(documents)} documents for user: {current_user}")
        return {"topics": topics, "documents": documents}
    except MySQLError as e:
        logger.error(f"Error listing contexts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing contexts: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.get("/user-history", response_model=UserHistoryResponse, dependencies=[Depends(get_current_user)])
async def get_user_history(current_user: str = Depends(get_current_user)):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Fetch topics searched by the user
        query_topics = """
            SELECT topic_name, instructions, created_at
            FROM topics
            WHERE user_id = (SELECT id FROM users WHERE username = %s)
            ORDER BY created_at DESC
        """
        cursor.execute(query_topics, (current_user,))
        topics = cursor.fetchall()
        
        # Fetch Q&A history
        query_qa = """
            SELECT question, answer, context_type, response_language, asked_at AS created_at
            FROM qa_history
            WHERE user_id = (SELECT id FROM users WHERE username = %s)
            ORDER BY asked_at DESC
        """
        cursor.execute(query_qa, (current_user,))
        qa_history = cursor.fetchall()
        
        logger.info(f"Retrieved {len(topics)} topics and {len(qa_history)} Q&A entries for user: {current_user}")
        return UserHistoryResponse(topics=topics, qa_history=qa_history)
    
    except MySQLError as e:
        logger.error(f"Error fetching user history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching user history: {str(e)}")
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.post("/generate-slides/topic", response_model=SlideResponse, dependencies=[Depends(get_current_user)])
async def generate_topic_slides(request: TopicRequest, current_user: str = Depends(get_current_user)):
    try:
        logger.info(f"Generating slides for topic: {request.topic}, user: {current_user}")
        check_slide_content_schema()
        slides, extra_explanations = generate_slides_from_topic(
            request.topic, request.instructions, request.num_slides, request.lang_choice
        )
        logger.info(f"Generated {len(slides)} slides, {len(extra_explanations)} explanations")
        
        if not isinstance(slides, list) or not all(isinstance(s, str) for s in slides):
            logger.error("Invalid slides format: not a list of strings")
            raise HTTPException(status_code=500, detail="Invalid slides format from generate_slides_from_topic")
        if not isinstance(extra_explanations, dict):
            logger.error("Invalid explanations format: not a dictionary")
            raise HTTPException(status_code=500, detail="Invalid explanations format from generate_slides_from_topic")
        
        ppt_filename = create_ppt(slides, filename_prefix=f"topic_{current_user}")
        ppt_path = os.path.join(PPT_DIR, ppt_filename)
        logger.info(f"Moving PPT to {ppt_path}")
        try:
            os.rename(ppt_filename, ppt_path)
        except OSError as e:
            logger.error(f"Failed to move PPT file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to move PPT file: {str(e)}")
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        query = "INSERT INTO topics (user_id, topic_name, instructions) VALUES ((SELECT id FROM users WHERE username = %s), %s, %s)"
        try:
            cursor.execute(query, (current_user, request.topic, request.instructions or ""))
            topic_id = cursor.lastrowid
            if not topic_id:
                raise HTTPException(status_code=400, detail="Failed to insert topic")
            logger.info(f"Inserted topic with ID: {topic_id}")
        except MySQLError as e:
            logger.error(f"Error inserting topic: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error inserting topic: {str(e)}")
        
        query = "INSERT INTO slides (user_id, topic_id, ppt_filename, slide_count) VALUES ((SELECT id FROM users WHERE username = %s), %s, %s, %s)"
        try:
            cursor.execute(query, (current_user, topic_id, ppt_filename, request.num_slides))
            slide_id = cursor.lastrowid
            if not slide_id:
                raise HTTPException(status_code=400, detail="Failed to insert slide")
            logger.info(f"Inserted slide with ID: {slide_id}")
        except MySQLError as e:
            logger.error(f"Error inserting slide: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error inserting slide: {str(e)}")
        
        for i, slide in enumerate(slides, 1):
            if slide.strip():
                title = slide.split("\n")[0].replace("## ", "").strip() if slide.split("\n") else "Untitled"
                content = "\n".join(slide.split("\n")[1:]).strip() if len(slide.split("\n")) > 1 else ""
                explanation = extra_explanations.get(f"Slide {i}", "")
                query = "INSERT INTO slide_content (slide_id, slide_number, title, content, explanation) VALUES (%s, %s, %s, %s, %s)"
                try:
                    cursor.execute(query, (slide_id, i, title, content, explanation))
                    logger.info(f"Inserted content for slide {i}")
                except MySQLError as e:
                    logger.error(f"Error inserting slide content for slide {i}: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error inserting slide content: {str(e)}")
        
        connection.commit()
        logger.info("Database operations completed successfully")
        
        return SlideResponse(slides=slides, explanations=extra_explanations, ppt_filename=ppt_filename)
    
    except Exception as e:
        logger.error(f"Error in generate_topic_slides: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating slides: {str(e)}")
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.post("/generate-slides/document", response_model=SlideResponse, dependencies=[Depends(get_current_user)])
async def generate_document_slides(
    file: UploadFile = File(...),
    request: DocumentRequest = Depends(),
    current_user: str = Depends(get_current_user)
):
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in [".pdf", ".txt"]:
        logger.error(f"Unsupported file format: {file_extension}")
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    try:
        check_slide_content_schema()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        logger.info(f"Saving uploaded file to {file_path}")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        text = extract_text_from_file(file_path)
        logger.info(f"Extracted text from {file.filename}, length: {len(text)}")
        slides, extra_explanations = generate_slides_from_document(
            text, request.instructions, request.num_slides, request.lang_choice
        )
        logger.info(f"Generated {len(slides)} slides, {len(extra_explanations)} explanations")
        
        if not isinstance(slides, list) or not all(isinstance(s, str) for s in slides):
            logger.error("Invalid slides format: not a list of strings")
            raise HTTPException(status_code=500, detail="Invalid slides format from generate_slides_from_document")
        if not isinstance(extra_explanations, dict):
            logger.error("Invalid explanations format: not a dictionary")
            raise HTTPException(status_code=500, detail="Invalid explanations format from generate_slides_from_document")
        
        ppt_filename = create_ppt(slides, filename_prefix=f"doc_{current_user}")
        ppt_path = os.path.join(PPT_DIR, ppt_filename)
        logger.info(f"Moving PPT to {ppt_path}")
        try:
            os.rename(ppt_filename, ppt_path)
        except OSError as e:
            logger.error(f"Failed to move PPT file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to move PPT file: {str(e)}")
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        query = "INSERT INTO documents (user_id, filename, instructions) VALUES ((SELECT id FROM users WHERE username = %s), %s, %s)"
        try:
            cursor.execute(query, (current_user, file.filename, request.instructions or ""))
            document_id = cursor.lastrowid
            if not document_id:
                raise HTTPException(status_code=400, detail="Failed to insert document")
            logger.info(f"Inserted document with ID: {document_id}")
        except MySQLError as e:
            logger.error(f"Error inserting document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error inserting document: {str(e)}")
        
        query = "INSERT INTO slides (user_id, document_id, ppt_filename, slide_count) VALUES ((SELECT id FROM users WHERE username = %s), %s, %s, %s)"
        try:
            cursor.execute(query, (current_user, document_id, ppt_filename, request.num_slides))
            slide_id = cursor.lastrowid
            if not slide_id:
                raise HTTPException(status_code=400, detail="Failed to insert slide")
            logger.info(f"Inserted slide with ID: {slide_id}")
        except MySQLError as e:
            logger.error(f"Error inserting slide: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error inserting slide: {str(e)}")
        
        for i, slide in enumerate(slides, 1):
            if slide.strip():
                title = slide.split("\n")[0].replace("## ", "").strip() if slide.split("\n") else "Untitled"
                content = "\n".join(slide.split("\n")[1:]).strip() if len(slide.split("\n")) > 1 else ""
                explanation = extra_explanations.get(f"Slide {i}", "")
                query = "INSERT INTO slide_content (slide_id, slide_number, title, content, explanation) VALUES (%s, %s, %s, %s, %s)"
                try:
                    cursor.execute(query, (slide_id, i, title, content, explanation))
                    logger.info(f"Inserted content for slide {i}")
                except MySQLError as e:
                    logger.error(f"Error inserting slide content for slide {i}: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error inserting slide content: {str(e)}")
        
        connection.commit()
        logger.info("Database operations completed successfully")
        os.remove(file_path)
        logger.info(f"Removed uploaded file: {file_path}")
        
        return SlideResponse(slides=slides, explanations=extra_explanations, ppt_filename=ppt_filename)
    
    except Exception as e:
        logger.error(f"Error in generate_document_slides: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

@app.post("/generate-quiz", response_model=QuizResponse, dependencies=[Depends(get_current_user)])
async def generate_quiz_endpoint(request: QuizRequest, current_user: str = Depends(get_current_user)):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Check if the slide exists and belongs to the user
        query = """
            SELECT ppt_filename, topic_id, document_id
            FROM slides
            WHERE ppt_filename = %s AND user_id = (SELECT id FROM users WHERE username = %s)
        """
        cursor.execute(query, (request.slide_filename, current_user))
        slide_data = cursor.fetchone()
        if not slide_data:
            logger.error(f"Slide not found: {request.slide_filename}")
            raise HTTPException(status_code=404, detail="Slide not found or not owned by user")
        
        # Determine context type (topic or document)
        context_type = "topic" if slide_data["topic_id"] else "document"
        
        # Fetch slide content
        query = """
            SELECT content
            FROM slide_content
            WHERE slide_id = (SELECT id FROM slides WHERE ppt_filename = %s)
            ORDER BY slide_number
        """
        cursor.execute(query, (request.slide_filename,))
        slides = [row["content"] for row in cursor.fetchall()]
        if not slides:
            logger.error(f"No content found for slide: {request.slide_filename}")
            raise HTTPException(status_code=404, detail="No slide content found")
        
        logger.info(f"Generating quiz for slide: {request.slide_filename}, context_type: {context_type}")
        quiz_content, quiz_filename = generate_quiz(slides, context_type, request.lang_choice)
        
        # Update slides table with quiz filename
        query = "UPDATE slides SET quiz_filename = %s WHERE ppt_filename = %s"
        try:
            cursor.execute(query, (quiz_filename, request.slide_filename))
            connection.commit()
            logger.info(f"Updated slides table with quiz_filename: {quiz_filename}")
        except MySQLError as e:
            logger.error(f"Error updating slides table with quiz_filename: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error updating slides table: {str(e)}")
        
        return QuizResponse(quiz_content=quiz_content, quiz_filename=quiz_filename)
    
    except Exception as e:
        logger.error(f"Error in generate_quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.post("/qa", response_model=QAResponse, dependencies=[Depends(get_current_user)])
async def answer_question(request: QARequest, current_user: str = Depends(get_current_user)):
    try:
        lang_choice = "roman" if request.lang_choice in ["roman", "roman urdu"] else "english"
        response_lang = "Roman Urdu" if lang_choice == "roman" else "English"
        system_prompt = f"""You are an experienced teacher. Answer accurately in {response_lang}, matching the question's style. 
        Use the provided context and extra explanations if available, and ensure the response is clear, concise, and engaging with real-life examples."""
        logger.info(f"Processing QA request: question='{request.question}', context_type='{request.context_type}', lang='{response_lang}'")

        if request.context_type == "topic" and request.context:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            query = """
                SELECT sc.content, sc.explanation
                FROM slide_content sc
                JOIN slides s ON sc.slide_id = s.id
                JOIN topics t ON s.topic_id = t.id
                WHERE t.user_id = (SELECT id FROM users WHERE username = %s)
                AND t.topic_name = %s
            """
            try:
                cursor.execute(query, (current_user, request.context))
                slides_data = cursor.fetchall()
                logger.info(f"Retrieved {len(slides_data)} slides for topic: {request.context}")
            except MySQLError as e:
                logger.error(f"Error querying slides: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error querying slides: {str(e)}")
            
            relevant_slides = []
            for slide in slides_data:
                if request.question.lower() in slide["content"].lower():
                    relevant_slides.append((slide["content"], slide["explanation"] or ""))
            
            if relevant_slides:
                context = "\n".join([f"Slide Content: {slide}\nTeacher Explanation: {exp}" for slide, exp in relevant_slides])
                logger.info(f"Found {len(relevant_slides)} relevant slides")
            else:
                context = "\n".join([f"Slide Content: {slide['content']}\nTeacher Explanation: {slide['explanation'] or ''}" for slide in slides_data])
                logger.info("No relevant slides found, using all slides as context")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{system_prompt}\nContext: {context}"),
                ("user", "Question: {question}"),
            ])
            chain = prompt | llm
            answer = chain.invoke({"question": request.question}).content
            cursor.close()
            connection.close()

        elif request.context_type == "document" and request.context:
            chunks = split_text_into_chunks(request.context)
            embeddings = generate_embeddings(chunks)
            index = create_faiss_index(embeddings)
            relevant_chunks = retrieve_relevant_chunks(request.question, index, chunks)
            logger.info(f"Retrieved {len(relevant_chunks)} relevant document chunks")
            if relevant_chunks:
                doc_explanation = generate_extra_explanation(
                    "\n".join(relevant_chunks), context="document", lang_choice=lang_choice
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"{system_prompt}\nContext: {relevant_chunks}\nTeacher Explanation: {doc_explanation}"),
                    ("user", "Question: {question}"),
                ])
                chain = prompt | llm
                answer = chain.invoke({"question": request.question}).content
            else:
                answer = "No relevant info found in the document." if response_lang == "English" else "Document mein is sawaal ka jawab nahi mila."
                logger.info("No relevant document chunks found")

        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "Question: {question}"),
            ])
            chain = prompt | llm
            answer = chain.invoke({"question": request.question}).content
            logger.info("No context provided, answering without context")

        connection = get_db_connection()
        cursor = connection.cursor()
        query = "INSERT INTO qa_history (user_id, question, answer, context_type, response_language) VALUES ((SELECT id FROM users WHERE username = %s), %s, %s, %s, %s)"
        try:
            cursor.execute(query, (current_user, request.question, answer, request.context_type, response_lang))
            connection.commit()
            logger.info("Inserted QA history")
        except MySQLError as e:
            logger.error(f"Error inserting QA history: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error inserting QA history: {str(e)}")
        finally:
            cursor.close()
            connection.close()

        return QAResponse(answer=answer)
    
    except Exception as e:
        logger.error(f"Error in answer_question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

@app.get("/slides/{filename}", dependencies=[Depends(get_current_user)])
async def download_slides(filename: str, current_user: str = Depends(get_current_user)):
    file_path = os.path.join(PPT_DIR, filename)
    if os.path.exists(file_path):
        logger.info(f"Downloading PPT: {file_path}")
        return FileResponse(
            file_path,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename=filename
        )
    logger.error(f"PPT file not found: {file_path}")
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/quiz/{filename}", dependencies=[Depends(get_current_user)])
async def download_quiz(filename: str, current_user: str = Depends(get_current_user)):
    file_path = os.path.join(QUIZ_DIR, filename)
    if os.path.exists(file_path):
        logger.info(f"Downloading quiz: {file_path}")
        return FileResponse(
            file_path,
            media_type="text/markdown",
            filename=filename
        )
    logger.error(f"Quiz file not found: {file_path}")
    raise HTTPException(status_code=404, detail="Quiz file not found")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    file_path = "C:/Users/uzmam/OneDrive/Documents/FYP/code/index.html"
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Trying to open file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            logger.info(f"Successfully read index.html (first 100 chars): {content[:100]}")
            return HTMLResponse(content=content, status_code=200)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="index.html not found")
    except Exception as e:
        logger.error(f"Error reading index.html: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading index.html: {str(e)}")

@app.get("/favicon.ico", response_class=FileResponse)
async def serve_favicon():
    favicon_path = "C:/Users/uzmam/OneDrive/Documents/FYP/code/favicon.ico"
    if os.path.exists(favicon_path):
        logger.info(f"Serving favicon: {favicon_path}")
        return FileResponse(favicon_path)
    logger.error("Favicon not found")
    raise HTTPException(status_code=404, detail="Favicon not found")