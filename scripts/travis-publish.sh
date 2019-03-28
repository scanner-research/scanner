#!/bin/bash

set -e

# Exit if this is not the master branch
if ! [ "$TRAVIS_BRANCH" = "master" -a "$TRAVIS_PULL_REQUEST" = "false" -a "$BUILD_TYPE" = "cpu" ]; then
    exit 0
fi

# Commit docs
REPO_PATH=git@github.com:scanner-research/scanner.git
HTML_PATH=build/docs/html
COMMIT_USER="Documentation Builder"
COMMIT_EMAIL="wcrichto@cs.stanford.edu"
CHANGESET=$(git rev-parse --verify HEAD)

# Install python package for autodoc
docker run $DOCKER_REPO:cpu-local /bin/bash -c "
cd /opt/scanner
git config --global user.name \"${COMMIT_USER}\"
git config --global user.email \"${COMMIT_EMAIL}\"

# Unencrypt ssh key
mkdir -p ~/.ssh/
chmod 0700 ~/.ssh/

openssl aes-256-cbc -K $encrypted_519f11e8a6d4_key -iv $encrypted_519f11e8a6d4_iv -in .travis/travisci_rsa.enc -out .travis/travisci_rsa -d
chmod 0600 .travis/travisci_rsa
cp .travis/travisci_rsa ~/.ssh/id_rsa
cp .travis/travisci_rsa.pub ~/.ssh/id_rsa.pub
chmod 0744 ~/.ssh/id_rsa.pub
ls -lah .travis
ls -lah ~/.ssh/

eval \`ssh-agent -s\`
ssh-add
rm -fr ~/.ssh/known_hosts
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

pip3 install twine
pip3 install Sphinx sphinx-autodoc-typehints
apt-get update && apt-get install -y doxygen

rm -rf ${HTML_PATH}
mkdir -p ${HTML_PATH}
git clone -b gh-pages ${REPO_PATH} --single-branch ${HTML_PATH}

cd ${HTML_PATH}
cp CNAME /tmp
git rm -rf .
cd -

cd docs
make html
cd -

cd build
cmake -D BUILD_DOCS=ON ..
make doxygen
cd -
cp -r build/doxygen/html ${HTML_PATH}/cpp

cd ${HTML_PATH}
cp /tmp/CNAME .
git add .
git commit -m \"Automated documentation build for changeset ${CHANGESET}.\"
git push origin gh-pages
cd -
"

# Tell Scannertools to rebuild
request_body='{
"request": {
"branch":"master"
}}'

curl -s -X POST \
   -H "Content-Type: application/json" \
   -H "Accept: application/json" \
   -H "Travis-API-Version: 3" \
   -H "Authorization: token $TRAVIS_TOKEN" \
   -d "$request_body" \
   "https://api.travis-ci.org/repo/scanner-research%2Fscannertools/requests"
