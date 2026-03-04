# n8n + SQLite (sin Microsoft)

Este flujo reemplaza `Microsoft Connect + LLM` por consulta directa a `db.sqlite3`.

## 1) Script disponible en este repo

Usa:

```bash
scripts/export_transactions_sqlite.sh <from_date> <to_date> [output_csv] [classification] [user_id] [db_path]
```

Ejemplo:

```bash
scripts/export_transactions_sqlite.sh 2026-01-01 2026-01-31 /tmp/gastos_enero.csv gasto 1 db.sqlite3
```

## 2) Flujo n8n recomendado

1. `Telegram Trigger`
2. `Set (From)`:
   - `From = {{$json.message.text}}` (formato esperado: `YYYY-MM-DD`)
3. `Set (Now)`:
   - `Now = {{$now.toUTC().startOf('day').minus({ days: 1 }).toISODate()}}`
4. `Execute Command`:
   - Command:

```bash
/Users/andresmauriciotrianareina/Documents/GitHub/Finance-App-AT/scripts/export_transactions_sqlite.sh \
  "{{$('From').item.json.From}}" \
  "{{$('Now').item.json.Now}}" \
  "/tmp/desde_{{$('From').item.json.From}}_hasta_{{$('Now').item.json.Now}}.csv" \
  "gasto" \
  "1" \
  "/Users/andresmauriciotrianareina/Documents/GitHub/Finance-App-AT/db.sqlite3"
```

5. `Read Binary File`:
   - File Path:

```text
/tmp/desde_{{$('From').item.json.From}}_hasta_{{$('Now').item.json.Now}}.csv
```

6. `Telegram` (Send Document) y/o `FTP` (upload)

## 3) Consulta SQL usada

```sql
SELECT
  t.bank,
  t.account_last4,
  t.date,
  t.time,
  t.amount,
  t.currency,
  COALESCE(mt.name, '') AS movement_type,
  t.classification,
  t.reference,
  t.merchant,
  t.location,
  t.channel,
  t.description,
  t.external_id AS id
FROM transactions_transaction t
LEFT JOIN transactions_movementtype mt
  ON mt.id = t.movement_type_id
WHERE t.user_id = 1
  AND t.date >= 'YYYY-MM-DD'
  AND t.date <= 'YYYY-MM-DD'
  AND t.classification = 'gasto'
ORDER BY t.date, t.time;
```

## 4) Verificacion en este proyecto

- Tabla usada: `transactions_transaction`
- Usuario encontrado: `user_id=1` (`andresmtr`)
- Rango de datos actual: `2020-01-02` a `2026-01-31`
