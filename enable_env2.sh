deactivate 2>/dev/null

rm -rf .venv
py -3.11 -m venv .venv
source .venv/Scripts/activate

which python
which pip
which langgraph

python -m pip install -U pip
python -m pip install --no-cache-dir "langgraph-cli[inmem]==0.3.7"
python -m pip install -e .
hash -r