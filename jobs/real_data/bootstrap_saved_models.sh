#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR/python"
"$PYTHON_BIN" -u -m real_data.bootstrap
