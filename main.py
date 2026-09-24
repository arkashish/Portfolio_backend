import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()

BASE_DIR = Path(__file__).parent
RESUME_PATH = BASE_DIR / "my_resume.pdf"
CACHE_PATH = BASE_DIR / ".resume_cache.json"


client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = "openai/gpt-oss-120b"

# Set ALLOWED_ORIGINS in .env for production, e.g.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")


# ---------- Schemas ----------
class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []


resume_schema = Resume.model_json_schema()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


# ---------- Resume loading ----------
def read_pdf(file_path: Path) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def parse_resume(resume_text: str) -> Resume:
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings, for example:
    Experience, Professional Experience, Work History, Employment, Internships.
    These may all contain relevant experience.

    Skills may appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:
    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse the following resume:\n\n{resume_text}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return Resume(**json.loads(response.choices[0].message.content))


def load_resume() -> Resume:
    """Parse the PDF once. Reuse the cached result until the PDF changes."""
    pdf_hash = hashlib.sha256(RESUME_PATH.read_bytes()).hexdigest()

    if CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text())
            if cached.get("hash") == pdf_hash:
                return Resume(**cached["resume"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # bad cache, re-parse below

    resume = parse_resume(read_pdf(RESUME_PATH))
    CACHE_PATH.write_text(json.dumps({"hash": pdf_hash, "resume": resume.model_dump()}))
    return resume


# ---------- App ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.resume = load_resume()  # runs once at startup
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def ask_candidate(question: str, resume: Resume) -> str:
    name = resume.name or "the candidate"
    system_prompt = f"""
    You are an AI assistant on {name}'s portfolio website. Recruiters and
    hiring managers ask you questions about {name}.

    Everything you know about {name}:

    {resume.model_dump_json(indent=2)}

    Rules:
    1. Answer only using the information above.
    2. Never make up facts, numbers, dates or employers.
    3. If the information is unavailable, say exactly:
       "I don't have enough information to answer that."
    4. Speak about {name} in the third person, in a professional and concise tone.
    5. Keep answers short (a few sentences) unless asked for detail.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


@app.get("/")
def home():
    return {"message": "Hello From basic FastApi Code", "resume_loaded": bool(app.state.resume)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = ask_candidate(request.question, app.state.resume)
    except Exception as e:
        print(f"Groq error: {e}")
        raise HTTPException(status_code=502, detail="The AI service is unavailable. Try again shortly.")
    return {"answer": answer}