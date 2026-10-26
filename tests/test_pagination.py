"""fetch_all: o Supabase corta respostas em 1000 linhas; a leitura deve paginar."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.pagination import fetch_all  # noqa: E402


class _FakeQuery:
    """Simula `.order().range().execute()` sobre uma tabela em memória."""

    def __init__(self, rows, cap):
        self.rows, self.cap = rows, cap
        self.calls = []
        self._lo = self._hi = None

    def order(self, _col):
        return self

    def range(self, lo, hi):
        self._lo, self._hi = lo, hi
        return self

    def execute(self):
        self.calls.append((self._lo, self._hi))
        hi = min(self._hi, self._lo + self.cap - 1)  # teto do servidor
        return type("R", (), {"data": self.rows[self._lo:hi + 1]})()


def test_le_todas_as_linhas_acima_do_teto():
    rows = [{"id": i} for i in range(2350)]
    q = _FakeQuery(rows, cap=1000)
    assert fetch_all(lambda: q) == rows
    assert q.calls == [(0, 999), (1000, 1999), (2000, 2999)]


def test_tabela_pequena_faz_uma_requisicao():
    q = _FakeQuery([{"id": 1}, {"id": 2}], cap=1000)
    assert len(fetch_all(lambda: q)) == 2
    assert len(q.calls) == 1


def test_tabela_vazia():
    assert fetch_all(lambda: _FakeQuery([], cap=1000)) == []
