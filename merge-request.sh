#!/bin/bash

set -e

git pull --rebase

VERSION=$(poetry version | awk 'END {print $NF}' | sed -e 's/-alpha./a/' -e 's/-beta./b/')
BRANCH="v${VERSION}"

git switch -c $BRANCH
git tag $VERSION

poetry lock
git add poetry.lock
git commit -m "bot: chg: updated poetry lockfile" --no-verify || echo "No new dependencies"

gitchangelog > changelog.rst
git add changelog.rst
git commit -m "bot: chg: updated changelog" --no-verify

git push origin $BRANCH \
        -o merge_request.create \
        -o merge_request.merge_when_pipeline_succeeds \
        -o merge_request.title=$VERSION \
        -o merge_request.description=$VERSION \
        -o merge_request.assign="marvin.vanaalst"
git push origin $VERSION

git switch main
git branch -D $BRANCH
