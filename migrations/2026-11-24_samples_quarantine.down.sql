-- =====================================================================
-- DOWN: desfaz 2026-11-24_samples_quarantine.sql
-- =====================================================================
--
-- Deixa o banco como estava antes: remove a tabela `samples_quarantine`.
--
-- Cuidado com o que esta' DENTRO dela. Ha' dois tipos de item:
--
--   (a) Movidos pela auditoria (scripts/audit_dataset.py --apply): ESTAVAM em
--       `samples` e foram tirados de la'. Para voltar ao estado anterior eles
--       precisam voltar para `samples` — este script faz isso por padrao
--       (restaurar_auditoria = true).
--
--   (b) Vindos da coleta: amostras suspeitas que NUNCA estiveram em
--       `samples`. Antes da migracao elas nem existiriam, entao apaga'-las
--       e' o estado anterior. Se preferir manter, mude restaurar_coleta
--       para true (elas entram no treino sem revisao).
--
-- Roda numa transacao: se qualquer passo falhar, nada muda.
--
-- PASSO 1 (recomendado) — veja o que existe antes de reverter:
--
--   select case when scores ? 'original_sample_id' then 'auditoria' else 'coleta' end as origem,
--          count(*)
--     from public.samples_quarantine group by 1;
--
-- PASSO 2 — rode o bloco abaixo.

begin;

do $$
declare
    restaurar_auditoria boolean := true;   -- (a) devolve para samples
    restaurar_coleta    boolean := false;  -- (b) false = descarta
    features_type text;
    features_expr text;
    filtro        text;
    restored      bigint;
begin
    if to_regclass('public.samples_quarantine') is null then
        raise notice 'samples_quarantine nao existe; nada a reverter.';
        return;
    end if;

    -- O tipo de samples.features decide a conversao do jsonb da quarentena.
    select format_type(a.atttypid, a.atttypmod) into features_type
      from pg_attribute a
     where a.attrelid = 'public.samples'::regclass and a.attname = 'features' and not a.attisdropped;

    features_expr := case
        when features_type = 'jsonb' then 'q.features'
        when features_type = 'json'  then 'q.features::json'
        when features_type like '%[]' then format(
            'array(select jsonb_array_elements_text(q.features))::%s', features_type)
        else null
    end;
    if features_expr is null then
        raise exception 'Tipo de samples.features nao suportado (%); nada foi alterado.', features_type;
    end if;

    filtro := case
        when restaurar_auditoria and restaurar_coleta then 'true'
        when restaurar_auditoria then $f$q.scores ? 'original_sample_id'$f$
        when restaurar_coleta    then $f$not (q.scores ? 'original_sample_id')$f$
        else 'false'
    end;

    -- A amostra volta com id e created_at NOVOS (os originais ficam em
    -- scores->>'original_sample_id'). Nada no app depende do id da amostra;
    -- created_at so' pesa na separacao de sessoes do teste do treino estatico.
    execute format($f$
        insert into public.samples (dataset_id, label, features, user_id)
        select q.dataset_id, q.label, %s, q.user_id
          from public.samples_quarantine q
         where %s
    $f$, features_expr, filtro);
    get diagnostics restored = row_count;
    raise notice '% amostra(s) devolvida(s) para samples.', restored;

    drop table public.samples_quarantine;
    raise notice 'samples_quarantine removida.';
end $$;

commit;
