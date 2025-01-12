#!/bin/bash

# Run these to initialize
# sudo apt update
# sudo apt install python3-pip python3-venv
# python3 -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt

function kill_port() {
    port=$1
    if [ "$(lsof -t -i:$port)" ]; then
        echo "Port $port is in use. Killing process..."
        kill -9 $(lsof -t -i:$port)
    else
        echo "Port $port not in use."
    fi
}

source .venv/bin/activate

source .env
cd server

version=$1 # 'dev' or 'prod'

if [ "$version" == "dev" ]; then
    # We use localhost because 127.0.0.1 doesn't work with google auth
    fastapi dev main.py --host localhost
elif [ "$version" == "prod" ]; then
    kill_port 443

    # Run in the background and write to log.txt
    nohup uvicorn main:app --host 0.0.0.0 --port 443 --ssl-keyfile=/etc/letsencrypt/live/gauntlet.spencerchubb.com/privkey.pem --ssl-certfile=/etc/letsencrypt/live/gauntlet.spencerchubb.com/fullchain.pem > log.txt 2>&1 &
else
    echo "Usage: ./run.sh [dev|prod]"
fi