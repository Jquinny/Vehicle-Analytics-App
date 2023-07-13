# Truck-Analytics-App

application for analyzing vehicles in cctv roadway footage, gathering vehicle data such as class, color, entry, and exit location

### Python Virtual Environment Setup

First, make sure you have a version of python >= 3.9.X (3.10 is what this has
been tested on so far)

1. run `python -m venv env` from the root directory of this repository. You should
   end up with an env/ directory inside the vehicle-analytics-app directory (`python3`
   if on linux or mac)
2. activate the environment by running `.\env\Scripts\activate.bat ` if on windows
   or `source env/bin/activate` if on linux or mac
3. run `python -m pip install -r requirements.txt` to install all dependencies
   into the virtual environment (MAKE SURE THE ENVIRONMENT IS ACTIVATED FIRST) (`python3`
   if on linux or mac)
