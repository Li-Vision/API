import logging
import time

logger = logging.getLogger(__name__)

# Fragmentos de mensagem de erros de conexao que o Supabase/httpx levanta quando
# uma conexao do pool foi encerrada do outro lado enquanto ficava ociosa.
_TRANSIENT_MARKERS = ("server disconnected", "connection reset", "connection closed",
                      "remoteprotocolerror", "readerror", "timed out")


def is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(m in text for m in _TRANSIENT_MARKERS)


def retry_transient(fn, attempts: int = 3, delay: float = 0.3):
    """
    Executa `fn()` repetindo se a falha for de conexao ociosa derrubada.

    So' deve embrulhar UMA operacao de banco por vez: repetir um bloco com
    varias escritas poderia duplicar a que ja' tinha dado certo.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts or not is_transient(e):
                raise
            logger.warning("Falha transitoria de conexao (%s); tentativa %d/%d", e, attempt, attempts)
            time.sleep(delay * attempt)
