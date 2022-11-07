#!/bin/bash

# Helper to setup a virtual environment with the latest requirements

PERFETTO_SRCDIR=${PERFETTO_SRCDIR:-~/git/cheri-perfetto}
TARGET_DIR=${TARGET_DIR:-venv}
WHICH_PYTHON=${WHICH_PYTHON:-/usr/bin/python3}

if [ "$1" = "-h" ]; then
    echo "Set the following env vars for control"
    echo "PERFETTO_SRCDIR [default: ~/git/cheri-perfetto]"
    echo "TARGET_DIR [default: ./venv]"
    echo "WHICH_PYTHON [default: /usr/bin/python3]"
    exit 0
fi

echo "Installing git hooks"
ln -s ../../git-hooks/pre-commit .git/hooks/pre-commit
ln -s ../../git-hooks/pre-push .git/hooks/pre-push

echo "Create virtualenv at $TARGET_DIR with $WHICH_PYTHON"

virtualenv --python $WHICH_PYTHON $TARGET_DIR

echo "Entering virtualenv"
source $TARGET_DIR/bin/activate

echo "Fetch updated pip"
curl -sS https://bootstrap.pypa.io/get-pip.py | python
pip install --upgrade setuptools

pip install -e .
pip install -e "${PERFETTO_SRCDIR}/src/trace_processor/python"

echo "Exiting virtualenv"
deactivate
