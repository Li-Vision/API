-- =====================================================================
-- UP: cria a quarentena de amostras suspeitas
-- Reverter: 2026-11-24_samples_quarantine.down.sql
-- =====================================================================
--
-- Amostras que passam na integridade mas destoam do dataset ficam aqui em vez
-- de `samples`: o treino le' apenas `samples`, entao nada daqui chega ao
-- modelo ate' um admin aprovar (POST /collect/quarantine/{id}/approve).
--
-- O que este script FAZ:   cria UMA tabela nova (+ indice) e liga o RLS nela.
-- O que ele NAO faz:       nao altera, nao le' e nao apaga nada em `samples`,
--                          `datasets` ou qualquer outra tabela existente.
--
-- Seguro rodar mais de uma vez (if not exists). Roda numa transacao: se
-- qualquer passo falhar, nada fica aplicado.
--
-- Como rodar: Supabase -> SQL Editor -> colar tudo -> Run.

begin;

-- Os tipos de dataset_id / user_id copiam os das tabelas existentes
-- (uuid, bigint...), para a chave estrangeira sempre casar.
do $$
declare
    dataset_id_type text;
    user_id_type    text;
begin
    select format_type(a.atttypid, a.atttypmod) into dataset_id_type
      from pg_attribute a
     where a.attrelid = 'public.datasets'::regclass and a.attname = 'id' and not a.attisdropped;

    select format_type(a.atttypid, a.atttypmod) into user_id_type
      from pg_attribute a
     where a.attrelid = 'public.samples'::regclass and a.attname = 'user_id' and not a.attisdropped;

    if dataset_id_type is null then
        raise exception 'public.datasets.id nao encontrado; nada foi criado.';
    end if;

    execute format($f$
        create table if not exists public.samples_quarantine (
            id          uuid primary key default gen_random_uuid(),
            dataset_id  %s not null references public.datasets(id) on delete cascade,
            label       text not null,
            kind        text not null check (kind in ('static', 'dynamic')),
            features    jsonb not null,
            user_id     %s,
            reasons     jsonb not null default '[]'::jsonb,   -- [{code, message, severity}]
            scores      jsonb not null default '{}'::jsonb,   -- {similarity, model, audit_status, original_sample_id}
            created_at  timestamptz not null default now()
        )$f$, dataset_id_type, coalesce(user_id_type, 'uuid'));
end $$;

create index if not exists samples_quarantine_dataset_label_idx
    on public.samples_quarantine (dataset_id, label);

-- O backend usa a service role (ignora RLS). RLS ligado sem politicas impede
-- que a chave anon do app leia ou grave aqui diretamente.
alter table public.samples_quarantine enable row level security;

commit;

-- Conferencia (opcional): deve listar as colunas da tabela nova.
-- select column_name, data_type from information_schema.columns
--  where table_schema = 'public' and table_name = 'samples_quarantine';
