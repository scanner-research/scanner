#!/bin/bash

openssl aes-256-cbc -K $encrypted_519f11e8a6d4_key -iv $encrypted_519f11e8a6d4_iv -in .config/travisci_rsa.enc -out .config/travisci_rsa -d
chmod 0600 .config/travisci_rsa
eval `ssh-agent -s`
ssh-add .config/travisci_rsa
