#!/bin/bash
papers=(
  "2510.09864"
  "2507.09307"
  "2510.02580"
  "2509.05171"
  "2509.12399"
  "2504.00817"
  "2002.02821"
  "1407.6387"
  "2403.03255"
  "1301.2826"
  "1511.00790"
  "2411.13258"
  "2509.06428"
  "2208.08590"
  "1704.05473"
  "0912.0255"
  "1412.4708"
  "1903.07706"
  "1808.06832"
  "2507.05853"
  "2509.07008"
)

for p in "${papers[@]}"; do
  echo "Downloading arXiv:${p}..."
  wget -q -nc -O "data/external/papers/arxiv_${p}.pdf" "https://arxiv.org/pdf/${p}.pdf" || echo "Failed to download ${p}"
done
echo "Downloads complete."
