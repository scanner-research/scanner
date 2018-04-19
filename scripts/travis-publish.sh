# Exit if this is not the master branch
if ! [ "$TRAVIS_BRANCH" = "master" -a "$TRAVIS_PULL_REQUEST" = "false" ]; then
    exit 0
fi

# Commit docs
REPO_PATH=git@github.com:scanner-research/scanner.git
HTML_PATH=build/doc/html
COMMIT_USER="Documentation Builder"
COMMIT_EMAIL="wcrichto@cs.stanford.edu"
CHANGESET=$(git rev-parse --verify HEAD)

openssl aes-256-cbc -K $encrypted_519f11e8a6d4_key -iv $encrypted_519f11e8a6d4_iv -in .travis/travisci_rsa.enc -out .travis/travisci_rsa -d
chmod 0600 .travis/travisci_rsa
eval `ssh-agent -s`
ssh-add .travis/travisci_rsa
cp .travis/travisci_rsa.pub ~/.ssh/id_rsa.pub


rm -rf ${HTML_PATH}
mkdir -p ${HTML_PATH}
git clone -b gh-pages ${REPO_PATH} --single-branch ${HTML_PATH}

cd ${HTML_PATH}
cp CNAME /tmp
git rm -rf .
cp /tmp/CNAME .
cd -

cd sphinx
make html
cd -

cd ${HTML_PATH}
git add .
git config user.name "${COMMIT_USER}"
git config user.email "${COMMIT_EMAIL}"
git commit -m "Automated documentation build for changeset ${CHANGESET}."
git push origin gh-pages
cd -

# Publish Python package if on a new tag
if [ -n "$TRAVIS_TAG" ];
then
    docker run $DOCKER_REPO:gpu /bin/bash -c "
pip3 install twine && \
python3 setup.py bdist_wheel && \
twine upload -u 'wcrichto' -p '${PYPI_PASS}' dist/*
"
fi
