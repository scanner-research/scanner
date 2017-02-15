REPO_PATH=git@github.com:scanner-research/scanner.git
HTML_PATH=build/doc/html
COMMIT_USER="Documentation Builder"
COMMIT_EMAIL="wcrichto@cs.stanford.edu"
CHANGESET=$(git rev-parse --verify HEAD)

openssl aes-256-cbc -K $encrypted_519f11e8a6d4_key -iv $encrypted_519f11e8a6d4_iv -in .config/travisci_rsa.enc -out .config/travisci_rsa -d
chmod 0600 .config/travisci_rsa
eval `ssh-agent -s`
ssh-add .config/travisci_rsa
cp .config/travisci_rsa.pub ~/.ssh/id_rsa.pub


rm -rf ${HTML_PATH}
mkdir -p ${HTML_PATH}
git clone -b gh-pages ${REPO_PATH} --single-branch ${HTML_PATH}

cd ${HTML_PATH}
git rm -rf .
cd -

doxygen .Doxyfile

cd ${HTML_PATH}
git add .
git config user.name "${COMMIT_USER}"
git config user.email "${COMMIT_EMAIL}"
git commit -m "Automated documentation build for changeset ${CHANGESET}."
git push origin gh-pages
cd -
