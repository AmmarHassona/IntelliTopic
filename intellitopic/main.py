from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    role: str = Form(...),
    mode: str = Form(...),
    input_text: str = Form(...)
):

    if mode == "research":
        prompt = f"""Generate 3 unique and comprehensive thesis topics for a {role} based on: {input_text}

For each thesis topic, provide:
1. A clear, descriptive title
2. A concise but detailed explanation of the research problem/objective (2-3 sentences)
3. The significance and potential impact of the research
4. Key research questions or gaps to explore
5. Suggested methodology or approach
6. Expected outcomes and contributions to the field

Format each topic with clear headings and bullet points for easy reading."""
    else:
        prompt = f"""Generate 3 creative and comprehensive course/module ideas for a {role} based on: {input_text}

For each course/module, provide:
1. A descriptive course title
2. A detailed course description and learning objectives (3-4 sentences)
3. Key topics and themes to be covered
4. Learning outcomes and skills students will develop
5. Suggested teaching methods and activities
6. Assessment strategies and evaluation criteria
7. Prerequisites or recommended background knowledge

Format each course with clear headings and bullet points for easy reading."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
        )
        generated = response.choices[0].message.content
    except Exception as e:
        generated = f"OpenAI API error: {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": generated,
        "input_text": input_text,
        "role": role,
        "mode": mode
    })