# main.py

from fastapi import FastAPI
# 1. Import the Solara FastAPI integration module
from solara.server import fastapi as solara_fastapi 

# 2. Create your main FastAPI application object
fast_app = FastAPI()

# 3. CRITICAL FIX: Include the Solara router in your FastAPI app
# This makes all Solara components available under the /solara path
fast_app.include_router(solara_fastapi.router)

# ... any other code, middleware, or routes you have ...

# You do not need (and should remove) any old code that looked like:
# fast_app = solara_fastapi.fast_app(fast_app)