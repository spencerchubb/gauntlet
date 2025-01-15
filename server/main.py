import asyncio
import base64
import json
from datetime import datetime
from typing import Annotated, List
import tempfile
import os

import boto3
import openai
from fastapi import Cookie, FastAPI, Request, Response, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fish_audio_sdk import Session as FishSession, TTSRequest
from fish_audio_sdk.schemas import Prosody
from pydantic import BaseModel, field_serializer
from sqlmodel import create_engine, delete, Field, func, select, Session, SQLModel, text, update

from completion import bedrock_completion
from rag import add_documents, similarity_search

fish_client = FishSession(apikey=os.getenv("FISH_KEY"))
openai_client = openai.OpenAI()

JwtCookie = Annotated[str | None, Cookie()]

class Channel(SQLModel, table=True):
    channel_id: int | None = Field(default=None, primary_key=True)
    name: str

class Dm(SQLModel, table=True):
    dm_id: int | None = Field(default=None, primary_key=True)
    uid: str | None = Field(default=None, primary_key=True)

class Message(SQLModel, table=True):
    message_id: int | None = Field(default=None, primary_key=True)

    # Either channel_id, dm_id, or thread_id must be set
    channel_id: int | None = Field(default=None)
    dm_id: int | None = Field(default=None)
    thread_id: int | None = Field(default=None)

    uid: str
    created: datetime | None = Field(default_factory=lambda: datetime.now())
    content: str
    reactions: str | None = Field(default=None)
    # vector_store_id: str
    
    @field_serializer("created")
    def serialize_created(self, created: datetime) -> str:
        return created.strftime("%m-%d %H:%M")
    
    @field_serializer("reactions")
    def serialize_reactions(self, reactions: str) -> dict[str, int]:
        return json.loads(reactions or "{}")

class User(SQLModel, table=True):
    uid: str = Field(primary_key=True)
    email: str
    name: str
    photo_url: str
    status: str | None = Field(default=None)
    ai_enabled: bool = Field(default=False)
    prompt: str | None = Field(default=None)

engine = create_engine("sqlite:///db.sqlite3")
SQLModel.metadata.create_all(engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gauntlet.spencerchubb.com", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

import firebase_admin
from firebase_admin import credentials, auth
cred = credentials.Certificate("firebase_service_key.json")
firebase_admin.initialize_app(cred)

@app.get("/")
def index(req: Request, jwt: JwtCookie = None, channel_id: int | None = None, dm_id: int | None = None, thread_id: int | None = None, query: str | None = None):
    try:
        uid = auth.verify_session_cookie(jwt, check_revoked=True)["uid"]
    except Exception as e:
        print(e)
        return templates.TemplateResponse(req, "signin.html")

    with Session(engine) as session:
        # Add mock data if it doesn't exist
        if not session.exec(select(User).where(User.uid == "gauntlet-bot")).first():
            session.add(User(uid="gauntlet-bot", name="Gauntlet Bot", email="gauntlet@gauntletai.com", photo_url="/static/gauntlet_logo.png"))
            session.commit()
        if not session.exec(select(User).where(User.name == "Kamala Harris")).first():
            session.add(User(uid="a", name="Kamala Harris", email="kamala@gauntletai.com", photo_url="https://pbs.twimg.com/profile_images/1592241313700782081/T2pTYU8d_400x400.jpg"))
            session.commit()
        if not session.exec(select(User).where(User.name == "Donald Trump")).first():
            session.add(User(uid="b", name="Donald Trump", email="donald@gauntletai.com", photo_url="https://pbs.twimg.com/profile_images/874276197357596672/kUuht00m_400x400.jpg"))
            session.commit()

        if query:
            messages = session.exec(select(Message, User).where(Message.content.contains(query)).join(User, User.uid == Message.uid)).all()
            messages = [{**message.model_dump(), **user.model_dump()} for message, user in messages]
            return templates.TemplateResponse(req, "search.html", {
                "query": query,
                "messages": messages,
            })

        # Use raw sql because it's hard to do with sqlmodel
        dms = session.exec(text(f"""
            select dm1.dm_id, user.name from dm as dm1
            join dm as dm2 on dm1.dm_id=dm2.dm_id and dm1.uid != dm2.uid
            join user on dm2.uid=user.uid
            where dm1.uid='{uid}'""")).all()
        dms = [{"dm_id": dm[0], "name": dm[1]} for dm in dms]

        channels = session.exec(select(Channel)).all()
        if not channels:
            # Create a "general" channel
            session.add(Channel(name="general"))
            session.commit()
            channels = session.exec(select(Channel)).all()

        current_channel = None
        current_dm = None
        current_thread = None
        messages = []
        if not channel_id and not dm_id and not thread_id:
            channel_id = channels[0].channel_id

        if channel_id:
            current_channel = session.exec(select(Channel).where(Channel.channel_id == channel_id)).first()
            if not current_channel:
                channel_id = channels[0].channel_id
                current_channel = session.exec(select(Channel).where(Channel.channel_id == channel_id)).first()
            messages = session.exec(select(Message, User).where(Message.channel_id == channel_id).join(User, User.uid == Message.uid)).all()
        elif dm_id:
            current_dm = session.exec(select(Dm).where(Dm.dm_id == dm_id).where(Dm.uid != uid)).first()
            current_dm = session.exec(select(User).where(User.uid == current_dm.uid)).first()
            current_dm = { "dm_id": dm_id, **current_dm.model_dump() }
            messages = session.exec(select(Message, User).where(Message.dm_id == dm_id).join(User, User.uid == Message.uid)).all()
        elif thread_id:
            current_thread = session.exec(select(Message).where(Message.message_id == thread_id)).first()
            messages = session.exec(select(Message, User).where(Message.thread_id == thread_id).join(User, User.uid == Message.uid)).all()
        users = session.exec(select(User).where(User.uid != uid)).all()
        current_user = session.exec(select(User).where(User.uid == uid)).first()

    messages = [{"message": message.model_dump(), "sender": user.model_dump()} for message, user in messages]
    return templates.TemplateResponse(
        req,
        "index.html",
        {
            "channels": [channel.model_dump() for channel in channels],
            "dms": dms,
            "current_channel": current_channel.model_dump() if current_channel else None,
            "current_dm": current_dm,
            "current_thread": current_thread.model_dump() if current_thread else None,
            "messages": messages,
            "users": [user.model_dump() for user in users],
            "current_user": current_user.model_dump() if current_user else None,
        },
    )

@app.get("/signin")
def signin(req: Request):
    return templates.TemplateResponse(req, "signin.html")

@app.get("/verify")
def verify(req: Request):
    return templates.TemplateResponse(req, "verify.html")

@app.get("/files")
def files(req: Request):
    files = []
    s3_client = boto3.client("s3", region_name="us-east-2")
    res = s3_client.list_objects_v2(Bucket="spencer-chubb-gauntlet", Prefix="chatgenius/")
    if "Contents" in res:
        for file in res["Contents"]:
            files.append(file["Key"].split("/")[1])
    return templates.TemplateResponse(req, "files.html", {"files": files})

websockets = {}

@app.websocket("/ws")
async def ws(websocket: WebSocket, jwt: JwtCookie = None):
    try:
        uid = auth.verify_session_cookie(jwt, check_revoked=True)["uid"]
    except Exception as e:
        return "Unauthorized", 401

    await websocket.accept()
    websockets[uid] = websocket
    print(f"connection opened for {uid} ({len(websockets)} total)")
    
    try:
        while True:
            data = await websocket.receive_json()
    except WebSocketDisconnect:
        if uid in websockets:
            del websockets[uid]
            print(f"connection closed for {uid} ({len(websockets)} total)")

class CreateChannelBody(BaseModel):
    name: str

@app.post("/channels/create")
def create_channel(req: Request, body: CreateChannelBody):
    with Session(engine) as session:
        session.add(Channel(name=body.name))
        session.commit()

class UpdateChannelBody(BaseModel):
    channel_id: int
    name: str

@app.post("/channels/update")
def update_channel(req: Request, body: UpdateChannelBody):
    with Session(engine) as session:
        session.exec(update(Channel).where(Channel.channel_id == body.channel_id).values(name=body.name))
        session.commit()

class DeleteChannelBody(BaseModel):
    channel_id: int

@app.post("/channels/delete")
def delete_channel(req: Request, body: DeleteChannelBody):
    with Session(engine) as session:
        session.exec(delete(Channel).where(Channel.channel_id == body.channel_id))
        session.exec(delete(Message).where(Message.channel_id == body.channel_id))
        session.commit()

class CreateDmBody(BaseModel):
    uid: str

@app.post("/dms/create")
def create_dm(req: Request, body: CreateDmBody, jwt: JwtCookie = None):
    try:
        my_uid = auth.verify_session_cookie(jwt, check_revoked=True)["uid"]
    except Exception as e:
        return "Unauthorized", 401

    with Session(engine) as session:
        dm_id = session.exec(select(func.max(Dm.dm_id))).first() or 0
        dm_id += 1
        dm = Dm(dm_id=dm_id, uid=body.uid)
        session.add(dm)
        dm = Dm(dm_id=dm_id, uid=my_uid)
        session.add(dm)
        session.commit()
        return {"dm_id": dm.dm_id}

# async def index_message(message: Message):
#     # TODO: Add document to vector store

#     # Save the id so we can associate regular messages with vector store
#     with(engine) as session:
#         message.vector_store_id = ids[0]
#         session.commit()

class CreateMessageBody(BaseModel):
    channel_id: int | None = None
    dm_id: int | None = None
    thread_id: int | None = None
    content: str

async def create_gauntlet_bot_message(body: CreateMessageBody) -> str:
    rag_output = answer_with_rag(body.content)

    with Session(engine) as session:
        uid = "gauntlet-bot"
        sender = session.exec(select(User).where(User.uid == uid)).first()
        message = Message(channel_id=body.channel_id, dm_id=body.dm_id, thread_id=body.thread_id, uid=uid, content=rag_output.answer)
        session.add(message)
        session.commit()
        session.refresh(message)
        session.refresh(sender)
        await send_websocket_messages(session, message, {"endpoint": "/messages/create", "message": message.model_dump(), "sender": sender.model_dump()})

async def create_ai_message(dm_id: int, uid: str):
    with Session(engine) as session:
        dm = session.exec(select(Dm).where(Dm.dm_id == dm_id).where(Dm.uid != uid)).first()
        sender = session.exec(select(User).where(User.uid == dm.uid)).first()
        if not sender.ai_enabled:
            return
        messages = session.exec(select(Message).where(Message.dm_id == dm_id)).all()
        system_prompt = f"Your job is to respond to Slack messages on behalf of {sender.name}. Be concise and to the point.\n\nHere is the prompt that {sender.name} has provided for you: {sender.prompt}"
        messages = [{"role": "user" if message.uid == uid else "assistant", "content": message.content} for message in messages]
        ai_response = bedrock_completion(system_prompt, messages)
        ai_response = Message(channel_id=None, dm_id=dm_id, thread_id=None, uid=sender.uid, content=ai_response)
        session.add(ai_response)
        session.commit()

        session.refresh(ai_response)
        session.refresh(sender)
        await send_websocket_messages(session, ai_response, {"endpoint": "/messages/create", "message": ai_response.model_dump(), "sender": sender.model_dump()})
        # asyncio.create_task(index_message(ai_response))
    
@app.post("/messages/create")
async def create_message(req: Request, body: CreateMessageBody, jwt: JwtCookie = None):
    try:
        uid = auth.verify_session_cookie(jwt, check_revoked=True)["uid"]
    except Exception as e:
        return "Unauthorized", 401

    with Session(engine) as session:
        sender = session.exec(select(User).where(User.uid == uid)).first()
        message = Message(channel_id=body.channel_id, dm_id=body.dm_id, thread_id=body.thread_id, uid=uid, content=body.content)
        session.add(message)

        session.commit()
        session.refresh(message)
        session.refresh(sender)

        payload = {"endpoint": "/messages/create", "message": message.model_dump(), "sender": sender.model_dump()}
        await send_websocket_messages(session, message, payload)
        # asyncio.create_task(index_message(message))
    
    if "@bot" in body.content:
        asyncio.create_task(create_gauntlet_bot_message(body))

    if body.dm_id:
        asyncio.create_task(create_ai_message(body.dm_id, uid))

class CreateReactionBody(BaseModel):
    message_id: int
    reaction: str

@app.post("/reactions/create")
async def create_reaction(req: Request, body: CreateReactionBody, jwt: JwtCookie = None):
    with Session(engine) as session:
        # Get message
        message = session.exec(select(Message).where(Message.message_id == body.message_id)).first()

        # Get reactions as dictionary
        reactions = json.loads(message.reactions or "{}")

        # Update reaction count
        reactions[body.reaction] = reactions.get(body.reaction, 0) + 1

        # Update message
        session.exec(update(Message).where(Message.message_id == body.message_id).values(reactions=json.dumps(reactions)))
        session.commit()

        payload = {"endpoint": "/reactions/create", "message_id": message.message_id, "reactions": reactions}
        await send_websocket_messages(session, message, payload)

class UpdateUserBody(BaseModel):
    status: str | None = None
    name: str | None = None
    photo_url: str | None = None
    ai_enabled: bool | None = None
    prompt: str | None = None

@app.post("/users/update")
def update_user(req: Request, body: UpdateUserBody, jwt: JwtCookie = None):
    try:
        uid = auth.verify_session_cookie(jwt, check_revoked=True)["uid"]
    except Exception as e:
        return "Unauthorized", 401

    with Session(engine) as session:
        session.exec(update(User).where(User.uid == uid).values(status=body.status, name=body.name, photo_url=body.photo_url, ai_enabled=body.ai_enabled, prompt=body.prompt))
        session.commit()

class AuthGoogleBody(BaseModel):
    uid: str
    email: str
    displayName: str
    photoUrl: str
    idToken: str

@app.post("/auth/google")
def auth_google(body: AuthGoogleBody, res: Response):
    user = User(
        uid=body.uid,
        email=body.email,
        name=body.displayName,
        photo_url=body.photoUrl,
    )

    with Session(engine) as session:
        existing_user = session.exec(select(User).where(User.uid == user.uid)).first()
        if existing_user:
            user.uid = existing_user.uid
        if not existing_user:
            session.add(user)
            session.commit()

    try:
        month = 1209600 # 2 weeks, maximum expiration allowed by the library
        jwt = auth.create_session_cookie(body.idToken, expires_in=month)
        res.set_cookie(key="jwt", value=jwt, expires=month)
        return
    except Exception as e:
        return "Unauthorized", 

@app.get("/generate_presigned_url")
def generate_presigned_url(req: Request, filename: str, method: str, jwt: JwtCookie = None):
    try:
        uid = auth.verify_session_cookie(jwt, check_revoked=True)["uid"]
    except Exception as e:
        return "Unauthorized", 401

    s3_client = boto3.client("s3", region_name="us-east-2")
    key = f"chatgenius/{filename}"
    url = s3_client.generate_presigned_url(
        f"{method.lower()}_object",
        Params={"Bucket": "spencer-chubb-gauntlet", "Key": key},
        ExpiresIn=3600,
        HttpMethod=method.upper(),
    )
    return {"url": url}

@app.post("/stt")
async def stt(file: UploadFile, jwt: JwtCookie = None):
    try:
        uid = auth.verify_session_cookie(jwt, check_revoked=True)["uid"]
    except Exception as e:
        return "Unauthorized", 401

    # Create a temporary file to store the audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file.flush()
        
        # Use the temporary file for transcription
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1", 
            file=open(temp_file.name, 'rb')
        )
    
    # Clean up the temporary file
    os.unlink(temp_file.name)
    return { "text": transcription.text }

class TtsBody(BaseModel):
    text: str

import nltk
from nltk.tokenize import PunktTokenizer
nltk.download("punkt_tab")
sentence_tokenizer = PunktTokenizer()

@app.post("/tts")
async def tts(body: TtsBody, jwt: JwtCookie = None):
    try:
        uid = auth.verify_session_cookie(jwt, check_revoked=True)["uid"]
    except Exception as e:
        return "Unauthorized", 401
    
    reference_ids = {
        "Austen Allred": "94b510e7818847e3804c5b4050d3c70d",
    }
    websocket = websockets[uid]

    sentences = sentence_tokenizer.tokenize(body.text.strip())
    import time
    start = time.time()
    first_audio = None
    for sentence in sentences:
        chunk_total = b""
        for chunk in fish_client.tts(TTSRequest(
            reference_id=reference_ids["Austen Allred"],
            text=sentence,
            format="wav",
            prosody=Prosody(speed=1.15),
        )):
            chunk_total += chunk
        if not first_audio:
            first_audio = time.time()
        audio_str = f"data:audio/wav;base64,{base64.b64encode(chunk_total).decode('utf-8')}"
        await websocket.send_json({"endpoint": "/tts", "audio": audio_str})

async def send_websocket_messages(session, message, payload):
    # If neither channel_id nor dm_id are populated, we're in a thread.
    # Traverse upward to find out if it's a channel or dm.
    while not message.channel_id and not message.dm_id:
        message = session.exec(select(Message).where(Message.message_id == message.thread_id)).first()

    if message.channel_id:
        # Notify everyone if it's a channel message
        for websocket in websockets.values():
            await websocket.send_json(payload)
    elif message.dm_id:
        # Notify users in the dm if it's a dm message
        for recipient in session.exec(select(Dm).where(Dm.dm_id == message.dm_id)).all():
            if recipient.uid not in websockets:
                continue
            await websockets[recipient.uid].send_json(payload)

class RagOutput(BaseModel):
    answer: str
    docs: List[str]

def answer_with_rag(question: str) -> str:
    hypothetical_document = bedrock_completion(
        """You are a question answering assistant for Gauntlet AI, an intensive AI training program for engineers.
Answer length MUST be 1 sentence to 1 paragraph in length. Answer questions with a decisive and convincing answer.
Do NOT express uncertainty, NEVER say you don't know something.
""",
        [{"role": "user", "content": question}],
        "llama3-2-3b",
    )
    hypothetical_document = f"\n{hypothetical_document}"

    context = ""
    docs = similarity_search(question + hypothetical_document)
    for i, doc in enumerate(docs):
        context += f"\n{i + 1} {doc}"
    prompt = f"""### Instructions
You are a question-answering assistant. You will be given a question and context.
For questions involving dates or times, give absolute answers instead of relative answers if possible (e.g. "3pm" instead of "in 2 hours").
Answer the question using the context provided.

### Question
{question}

### Context
{context}

### Answer
"""
    answer = bedrock_completion(
        "You are a question-answering assistant.",
        [{"role": "user", "content": prompt}],
        "llama3-3-70b",
    )
    return RagOutput(answer=answer, docs=docs)