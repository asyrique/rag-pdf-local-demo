## Instructions to run
1. Set up a Python installation on your machine
1. Install [Ollama](https://ollama.com/) and pull down:
  1. llama2:7b-chat (3.8GB)
  1. llama2-uncensored (3.8GB)
  1. mistral (4.1GB)
1. Initialize project by running `poetry install --no-root`
1. Run `poetry shell` to switch into the virtualenv created by Poetry
1. Run the project by running `streamlit run chat.py`