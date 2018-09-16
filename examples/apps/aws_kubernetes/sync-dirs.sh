#!/usr/local/bin/bash

DEFAULT_SERVER=localhost
DEFAULT_PORT=8022

LOCAL_DIR=$1
REMOTE_DIR=$2
REMOTE_SERVER=${3:-$DEFAULT_SERVER}
SSH_KEY=${4}

CMD="rsync -avz -e \"ssh -i $SSH_KEY\" \
          --exclude build \
          --exclude .git* \
          --exclude \#* \
          -r $LOCAL_DIR/ \
          $REMOTE_SERVER:$REMOTE_DIR"
eval $CMD

# inotifywait, linux
while fswatch -r $LOCAL_DIR/* -1; do
    eval $CMD || break;
done
