"""Leitura paginada de consultas do Supabase."""

# O PostgREST do Supabase devolve no máximo 1000 linhas por requisição; acima
# disso a resposta é cortada em silêncio.
SUPABASE_PAGE_SIZE = 1000


def fetch_all(make_query, page_size: int = SUPABASE_PAGE_SIZE, order_by: str = "id") -> list:
    """
    Lê TODAS as linhas de uma consulta, paginando com `.range()`.

    `make_query` é uma função sem argumentos que devolve uma consulta nova
    (ex.: `lambda: supabase.table("samples").select("label").eq(...)`); cada
    página usa uma consulta própria. A ordenação por `order_by` mantém as
    páginas consistentes entre as requisições.
    """
    rows: list = []
    offset = 0
    while True:
        page = make_query().order(order_by).range(offset, offset + page_size - 1).execute().data
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size
