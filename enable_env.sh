# rm -rf .venv

# python -3.11 -m venv .venv
source .venv/Scripts/activate

/d/_tina/learning/AI_project/retrieval-agent-template/.venv/Scripts/python -m pip install -U pip
/d/_tina/learning/AI_project/retrieval-agent-template/.venv/Scripts/python -m pip install --no-cache-dir "langgraph-cli[inmem]==0.3.8"
/d/_tina/learning/AI_project/retrieval-agent-template/.venv/Scripts/python -m pip install -e .
/d/_tina/learning/AI_project/retrieval-agent-template/.venv/Scripts/python -m pip install --no-cache-dir --force-reinstall "langgraph-api>=0.4.27" "langgraph-runtime-inmem>=0.14.0"


export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

pip list | grep langgraph
# langgraph dev