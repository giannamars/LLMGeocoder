import os
import getpass

from cborg_loader import init_cborg_chat_model

# ----------------------------------------------------------------------
# Initialise LLM
# ----------------------------------------------------------------------
if "CBORG_API_KEY" not in os.environ:
    os.environ["CBORG_API_KEY"] = getpass.getpass("Enter your CBorg API key: ")

llm = init_cborg_chat_model(model="lbl/cborg-chat:latest")


