# main.py

from fastapi import FastAPI
from solara.server import fastapi as solara_fastapi

# 1. CRITICAL: Import your Solara application file. 
# This action allows Solara to automatically discover and register 
# the components (like @sl.component def Page()) defined in app_solara.py.
import app_solara

# 2. Create the main FastAPI application object
fast_app = FastAPI()

# 3. CRITICAL FIX: Use the Solara app() wrapper function.
# This function modifies the 'fast_app' object, attaching all necessary 
# Solara endpoints (e.g., WebSockets, static assets) to your FastAPI router.
fast_app = solara_fastapi.app(fast_app)

# 4. (Optional) Define a simple health check or custom FastAPI route
@fast_app.get("/health")
async def health_check():
    """Simple endpoint to confirm the FastAPI server is running."""
    return {"status": "ok", "app": "SA_Drought_Bulletin (Solara + FastAPI)"}

# Uvicorn will run the 'fast_app' object defined above.
