#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/export_transactions_sqlite.sh <from_date> <to_date> [output_csv] [classification] [user_id] [db_path]
# Example:
#   scripts/export_transactions_sqlite.sh 2026-01-01 2026-01-31 /tmp/gastos.csv gasto 1 db.sqlite3

FROM_DATE="${1:-}"
TO_DATE="${2:-}"
OUTPUT_CSV="${3:-}"
CLASSIFICATION="${4:-gasto}"
USER_ID="${5:-1}"
DB_PATH="${6:-db.sqlite3}"

if [[ -z "${FROM_DATE}" || -z "${TO_DATE}" ]]; then
  echo "Error: from_date and to_date are required. Format: YYYY-MM-DD" >&2
  exit 1
fi

if [[ ! "${FROM_DATE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Error: from_date must use YYYY-MM-DD." >&2
  exit 1
fi

if [[ ! "${TO_DATE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Error: to_date must use YYYY-MM-DD." >&2
  exit 1
fi

if [[ ! "${USER_ID}" =~ ^[0-9]+$ ]]; then
  echo "Error: user_id must be numeric." >&2
  exit 1
fi

if [[ -z "${OUTPUT_CSV}" ]]; then
  OUTPUT_CSV="/tmp/transacciones_${FROM_DATE}_a_${TO_DATE}.csv"
fi

if [[ ! -f "${DB_PATH}" ]]; then
  echo "Error: SQLite file not found at ${DB_PATH}" >&2
  exit 1
fi

SQL="
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
WHERE t.user_id = ${USER_ID}
  AND t.date >= '${FROM_DATE}'
  AND t.date <= '${TO_DATE}'
"

if [[ -n "${CLASSIFICATION}" && "${CLASSIFICATION}" != "all" ]]; then
  SQL="${SQL}  AND t.classification = '${CLASSIFICATION}'"$'\n'
fi

SQL="${SQL}ORDER BY t.date, t.time;"

sqlite3 -header -csv "${DB_PATH}" "${SQL}" > "${OUTPUT_CSV}"

echo "CSV generated: ${OUTPUT_CSV}"
