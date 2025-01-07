import json
from datetime import datetime
from typing import Annotated

from fastapi import Cookie, FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_serializer
from sqlmodel import create_engine, delete, Field, select, Session, SQLModel, update

JwtCookie = Annotated[str | None, Cookie()]

class Channel(SQLModel, table=True):
    channel_id: int | None = Field(default=None, primary_key=True)
    name: str

class Message(SQLModel, table=True):
    message_id: int | None = Field(default=None, primary_key=True)

    # Either channel_id, dm_id, or thread_id must be set
    channel_id: int | None = Field(default=None)
    dm_id: str | None = Field(default=None)
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
def index(req: Request, jwt: JwtCookie = None, channel_id: int | None = None, dm_id: str | None = None, thread_id: int | None = None, query: str | None = None):
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
                messages = session.exec(select(Message).where(Message.content.contains(query))).all()
                return templates.TemplateResponse(req, "search.html", {
                    "query": query,
                    "messages": [message.model_dump() for message in messages],
                })

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
                messages = session.exec(select(Message).where(Message.channel_id == channel_id)).all()
            elif dm_id:
                current_dm = session.exec(select(User).where(User.uid == dm_id)).first()
                messages = session.exec(select(Message).where(Message.dm_id == dm_id)).all()
            elif thread_id:
                current_thread = session.exec(select(Message).where(Message.message_id == thread_id)).first()
                messages = session.exec(select(Message).where(Message.thread_id == thread_id)).all()
            users = session.exec(select(User).where(User.uid != uid)).all()
            current_user = session.exec(select(User).where(User.uid == uid)).first()

        return templates.TemplateResponse(
            req,
            "index.html",
            {
                "channels": [channel.model_dump() for channel in channels],
                "current_channel": current_channel.model_dump() if current_channel else None,
                "current_dm": current_dm.model_dump() if current_dm else None,
                "current_thread": current_thread.model_dump() if current_thread else None,
                "messages": [message.model_dump() for message in messages],
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

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
    except WebSocketDisconnect:
        print("WebSocket disconnected")

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

class CreateMessageBody(BaseModel):
    channel_id: int | None = None
    dm_id: str | None = None
    thread_id: int | None = None
    content: str

@app.post("/messages/create")
def create_message(req: Request, body: CreateMessageBody, jwt: JwtCookie = None):
    try:
        uid = auth.verify_session_cookie(jwt, check_revoked=True)["uid"]
    except Exception as e:
        return "Unauthorized", 401

    with Session(engine) as session:
        session.add(Message(channel_id=body.channel_id, dm_id=body.dm_id, thread_id=body.thread_id, uid=uid, content=body.content))
        session.commit()

class UpdateMessageBody(BaseModel):
    message_id: int
    content: str

@app.post("/messages/update")
def update_message(req: Request, body: UpdateMessageBody):
    with Session(engine) as session:
        session.exec(update(Message).where(Message.message_id == body.message_id).values(content=body.content))
        session.commit()

class DeleteMessageBody(BaseModel):
    message_id: int

@app.post("/messages/delete")
def delete_message(req: Request, body: DeleteMessageBody):
    with Session(engine) as session:
        session.exec(delete(Message).where(Message.message_id == body.message_id))
        session.commit()

class CreateReactionBody(BaseModel):
    message_id: int
    reaction: str

@app.post("/reactions/create")
def create_reaction(req: Request, body: CreateReactionBody, jwt: JwtCookie = None):
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

class UpdateUserBody(BaseModel):
    status: str | None = None

@app.post("/users/update")
def update_user(req: Request, body: UpdateUserBody, jwt: JwtCookie = None):
    try:
        uid = auth.verify_session_cookie(jwt, check_revoked=True)["uid"]
    except Exception as e:
        return "Unauthorized", 401

    with Session(engine) as session:
        session.exec(update(User).where(User.uid == uid).values(status=body.status))
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
            user.created = existing_user.created
        if not existing_user:
            session.add(user)
            session.commit()

    try:
        month = 1209600 # 2 weeks, maximum expiration allowed by the library
        jwt = auth.create_session_cookie(body.idToken, expires_in=month)
        res.set_cookie(key="jwt", value=jwt, expires=month)
        return
    except Exception as e:
        return "Unauthorized", 401