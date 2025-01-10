import asyncio
import json
from datetime import datetime
from typing import Annotated

import boto3
from fastapi import Cookie, FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_serializer
from sqlmodel import create_engine, delete, Field, func, select, Session, SQLModel, text, update

from completion import bedrock_completion

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
    reactions: str
    
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
    allow_origins=["https://pandapal.net", "https://pandapal.app", "http://localhost", "http://localhost:8000"],
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

        with Session(engine) as session:
            # Add mock data if it doesn't exist
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
            elif channel_id:
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
    except Exception as e:
        print(e)
        return templates.TemplateResponse(req, "signin.html")

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
# https://pbs.twimg.com/profile_images/1608281295918096385/D2kh-M28_400x400.jpg
# Austen Allred is the co-founder and CEO of BloomTech. A native of Springville, Utah, Austen’s start-up journey began in 2017 with him living in his two-door Civic while participating in Y Combinator, a San Francisco-based seed accelerator. This experience became the foundation of BloomTech’s rapid growth. Before founding BloomTech, Austen was the co-founder of media platform GrassWire. He co-authored the growth hacking textbook Secret Sauce, which became a best-seller and provided him the personal seed money to build BloomTech. Austen’s disruptive ideas on the future of education, the labor market disconnect, and the opportunity of providing opportunity at-scale have been featured in: The Harvard Business Review, The Economist, WIRED, Fast Company, TechCrunch, The New York Times, among others. Austen is fluent in Russian and currently lives in San Francisco with his wife and two kids. You can find him on Twitter @Austen.
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

async def create_ai_message(dm_id: int, uid: str):
    print("create_ai_message", dm_id, uid)
    with Session(engine) as session:
        dm = session.exec(select(Dm).where(Dm.dm_id == dm_id).where(Dm.uid != uid)).first()
        sender = session.exec(select(User).where(User.uid == dm.uid)).first()
        if not sender.ai_enabled:
            return
        print("The user has AI enabled!")
        messages = session.exec(select(Message).where(Message.dm_id == dm_id)).all()
        system_prompt = f"Your job is to respond to Slack messages on behalf of {sender.name}. Be concise and to the point.\n\nHere is the prompt that {sender.name} has provided for you: {sender.prompt}"
        messages = [{"role": "user" if message.uid == uid else "assistant", "content": message.content} for message in messages]
        ai_response = bedrock_completion(system_prompt, messages)
        ai_response = ai_response.replace("’", "'") # Replace weird apostrophes so it doesn't break JSON
        ai_response = Message(channel_id=None, dm_id=dm_id, thread_id=None, uid=sender.uid, content=ai_response)
        session.add(ai_response)
        session.commit()

        session.refresh(ai_response)
        session.refresh(sender)
        await send_websocket_messages(session, ai_response, {"endpoint": "/messages/create", "message": ai_response.model_dump(), "sender": sender.model_dump()})
    
class CreateMessageBody(BaseModel):
    channel_id: int | None = None
    dm_id: int | None = None
    thread_id: int | None = None
    content: str
    
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

    if body.dm_id:
        # Run without blocking the above websockets
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