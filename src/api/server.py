import logging

from fastapi import FastAPI

from src.api.app_state import state

from src.api.routes.admin_routes import router as admin_router
from src.api.routes.detect_routes import router as detect_router
from src.api.routes.collect_routes import router as collect_router
from src.api.routes.train_routes import router as train_router
from src.api.routes.websocket_routes import router as ws_router
from src.api.routes.auth_routes import router as auth_router
from src.api.routes.learning_routes import router as learning_router

# Sem isto os logger.info() da aplicacao (treino, coleta) nao aparecem no Render:
# o root logger fica em WARNING e so' o access log do uvicorn chega ao console.
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

app = FastAPI(title="Li-Vision API")

state.start_pipeline()
state.training_service.fail_orphaned_jobs()
# NOTA: Detectores agora são criados por sessão (UserSession).
# O ModelCache já foi inicializado no AppState.__init__.

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(detect_router)
app.include_router(collect_router)
app.include_router(train_router)
app.include_router(ws_router)
app.include_router(learning_router)


@app.get("/healthz")
def health_check():
    return {"status": "ok"}
