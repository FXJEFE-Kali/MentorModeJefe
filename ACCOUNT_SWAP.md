# Migrating to a new MT5 account / terminal

When you sign up at a new prop firm, install MT5, or move the project to a different machine, **everything that has to change is captured by these four fields**:

| Field | Where it lives | Typical value |
|---|---|---|
| `username` | Windows user (path prefix `C:\Users\<username>\…`) | `localonly` |
| `mt5_terminal_id` | 32-char hex in `…\Terminal\<ID>\…` | `D0E8209F77C8CF37AD8BF550E51FF075` |
| `broker` | The broker name MT5 logs (in EA logs and trade comments) | `FundingPips2-SIM` |
| `account_login` | Numeric account number | `12134397` |

The current values are stored in [`config.json`](config.json) under the top-level `_account` block.

## How to find the new values

After installing/logging-in to MT5 on the new account:
1. **Terminal ID** — open MT5 → File → Open Data Folder. Path looks like `C:\Users\<you>\AppData\Roaming\MetaQuotes\Terminal\<32-CHAR-HEX>\`. The hex is the terminal ID.
2. **Username** — `whoami` in PowerShell, or `$env:USERNAME`.
3. **Broker / login** — visible in MT5's Tools → Options → Server, or in the Navigator panel header.

## Migrating the project (one command)

Always preview first:

```powershell
python Scripts\swap_account.py --terminal-id NEW_HEX_ID --username NEW_USER --dry-run
```

When the diff looks right, drop `--dry-run`:

```powershell
python Scripts\swap_account.py --terminal-id NEW_HEX_ID --username NEW_USER --broker NEW_BROKER --login NEW_LOGIN
```

The script:
- Reads the current account from `config.json._account`.
- Replaces every occurrence in `config.json`, every `*.mq5` (for EA hardcoded paths), and every `Scripts/*.py` fallback.
- Writes the new values back into `config.json._account` so the next migration is idempotent.

## Manual checklist after running the swap

1. Validate `config.json`: `python -c "import json; json.load(open('config.json'))"` — should print nothing if valid.
2. **Recompile EAs** — open every modified `.mq5` in MetaEditor and press `F7`. The script doesn't touch compiled `.ex5` binaries.
3. **Restart the AI server** — `Stop-Process` whatever holds port 8080, then `python ai_server_golden.py`.
4. **Test the full pipeline**: `python pipelinerun.py` → expect 11/11 scripts pass.
5. **Test /predict**: `Invoke-RestMethod http://127.0.0.1:8080/health` → confirms the new account loaded.
6. **Verify EA on chart** — drag the EA onto a symbol in the new MT5 terminal, confirm it logs `Add to MT5 allowed URLs: …` cleanly.

## What the script will NOT touch

- `.venv\` — the Python virtual environment. Recreate fresh on a new machine: `python -m venv .venv && .venv\Scripts\Activate.ps1 && pip install -r FXJEFE_Institutional_ML_Stack.txt` (if pinned set exists) or fall back to `pip install pandas numpy scikit-learn xgboost lightgbm onnxruntime joblib MetaTrader5 pyzmq textblob optuna pandas-ta requests flask`.
- Existing log files (`Logs/*.log`).
- Model files (`*.pkl`, `*.onnx`, `*.h5`, `*.json`).
- The MT5 install itself — that's a manual install + login step.

## Rolling back

The previous values stay in `_account` only after you migrate. If you need to revert:

```powershell
python Scripts\swap_account.py --terminal-id OLD_HEX_ID --username OLD_USER
```
