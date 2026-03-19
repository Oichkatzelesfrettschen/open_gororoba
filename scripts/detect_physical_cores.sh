#!/bin/sh

set -eu

exec sh "$(dirname "$0")/detect_worker_budget.sh"
