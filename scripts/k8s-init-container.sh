#!/usr/bin/env bash
set -e

ADAPTER_PATH="${ADAPTER_PATH:-/mnt/adapters/latest}"
TARGET_ENGINE="${TARGET_ENGINE:-vllm}"
AUTO_FIX="${AUTO_FIX:-true}"

echo "==> [AdapterBridge K8s Init-Container] Validating mounted adapter volume at: ${ADAPTER_PATH}"
echo "==> Target Engine: ${TARGET_ENGINE}"

if [ ! -d "${ADAPTER_PATH}" ]; then
  echo "❌ Error: Adapter path '${ADAPTER_PATH}' does not exist!"
  exit 1
fi

if adapterbridge check --path "${ADAPTER_PATH}" --target "${TARGET_ENGINE}"; then
  echo "✔ Checkpoint validated successfully. Proceeding with pod launch."
  exit 0
else
  echo "⚠️ Validation failed."
  if [ "${AUTO_FIX}" = "true" ]; then
    echo "==> Executing automated remediation repair..."
    adapterbridge fix --src "${ADAPTER_PATH}" --dst "${ADAPTER_PATH}-repaired" --target "${TARGET_ENGINE}"
    echo "✔ Repaired adapter created at ${ADAPTER_PATH}-repaired. Container ready."
    exit 0
  else
    echo "❌ Validation failed and AUTO_FIX is set to false. Blocking pod launch."
    exit 1
  fi
fi
