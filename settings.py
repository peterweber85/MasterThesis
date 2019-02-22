#%% settings.py
from os.path import join, dirname
from dotenv import load_dotenv
import os

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

GMAPS_API_KEY = os.environ.get("GMAPS_API_KEY")

#%%