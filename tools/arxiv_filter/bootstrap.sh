#!/usr/bin/env bash
#
# Bootstrap an EC2 instance for the arxiv source filter pipeline.
#
# Idempotent: safe to re-run. Designed for Amazon Linux 2023 or Ubuntu 22.04+.
#
# Assumes the instance has the `arxiv-filter-ec2` IAM instance profile attached
# — no access keys to manage. Pulls filter.py + drive.py from the staging
# bucket (uploaded ahead of time from your dev machine).
#
# Run as:  sudo bash bootstrap.sh
# Or as user-data on instance launch.

set -euo pipefail

STAGING_BUCKET="${STAGING_BUCKET:-christian-arxiv-staging}"
PANDOC_VERSION="${PANDOC_VERSION:-3.6}"
WORK_USER="${WORK_USER:-$(logname 2>/dev/null || echo ec2-user)}"
WORK_HOME="$(getent passwd "$WORK_USER" | cut -d: -f6)"
INSTALL_DIR="$WORK_HOME/arxiv_filter"
SHARDS="${SHARDS:-4}"                 # number of concurrent drive.py instances
AUTO_RUN="${AUTO_RUN:-1}"             # 0 to skip auto-launch (manual mode)

echo "[bootstrap] user=$WORK_USER home=$WORK_HOME install_dir=$INSTALL_DIR"
echo "[bootstrap] staging bucket: $STAGING_BUCKET"

# 1. Detect OS family
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS_FAMILY="${ID_LIKE:-$ID}"
else
    OS_FAMILY="unknown"
fi
echo "[bootstrap] OS family: $OS_FAMILY ($PRETTY_NAME)"

# 2. System packages
# AL2023 ships curl-minimal already; installing 'curl' conflicts. Pip is
# included in python3 on AL2023 via the python3-pip package.
if [[ "$OS_FAMILY" == *"fedora"* || "$ID" == "amzn" ]]; then
    dnf install -y python3 python3-pip zstd tar unzip 2>&1 | tail -5
elif [[ "$OS_FAMILY" == *"debian"* || "$ID" == "ubuntu" ]]; then
    apt-get update
    apt-get install -y python3 python3-pip zstd tar curl unzip 2>&1 | tail -5
else
    echo "[bootstrap] unknown OS — install python3/pip/zstd/tar/curl/unzip manually" >&2
    exit 1
fi

# 3. Ensure aws-cli v2 is present (AL2023 has it; Ubuntu often does not)
if ! command -v aws >/dev/null; then
    echo "[bootstrap] installing aws-cli v2"
    cd /tmp
    curl -fLso awscliv2.zip "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
    unzip -q awscliv2.zip
    ./aws/install
fi
aws --version

# 4. Install pandoc 3.x (version-pinned to match dev environment)
need_install=true
if command -v pandoc >/dev/null; then
    current="$(pandoc --version | head -1 | awk '{print $2}')"
    [[ "$current" == "$PANDOC_VERSION"* ]] && need_install=false
fi
if $need_install; then
    echo "[bootstrap] installing pandoc $PANDOC_VERSION"
    cd /tmp
    curl -fLso pandoc.tar.gz \
        "https://github.com/jgm/pandoc/releases/download/$PANDOC_VERSION/pandoc-$PANDOC_VERSION-linux-amd64.tar.gz"
    tar xf pandoc.tar.gz
    install -m 0755 "pandoc-$PANDOC_VERSION/bin/pandoc" /usr/local/bin/pandoc
fi
pandoc --version | head -1

# 5. Python deps (zstandard for compressed-tar streaming writes)
python3 -m pip install --break-system-packages zstandard 2>&1 | tail -3 || \
    python3 -m pip install zstandard 2>&1 | tail -3

# 6. Pull filter.py + drive.py + watchdog.sh from staging
mkdir -p "$INSTALL_DIR"
aws s3 cp "s3://$STAGING_BUCKET/scripts/filter.py"   "$INSTALL_DIR/filter.py"
aws s3 cp "s3://$STAGING_BUCKET/scripts/drive.py"    "$INSTALL_DIR/drive.py"
aws s3 cp "s3://$STAGING_BUCKET/scripts/watchdog.sh" "$WORK_HOME/watchdog.sh"
chown -R "$WORK_USER:$WORK_USER" "$INSTALL_DIR" "$WORK_HOME/watchdog.sh"
chmod +x "$INSTALL_DIR"/*.py "$WORK_HOME/watchdog.sh"
echo "[bootstrap] installed scripts: $(ls "$INSTALL_DIR") watchdog.sh"

# 7. Verify AWS access via instance metadata
echo "[bootstrap] verifying AWS access via instance profile..."
aws sts get-caller-identity --query 'Arn' --output text
aws s3 ls "s3://$STAGING_BUCKET/" >/dev/null
aws s3api head-object --bucket arxiv --key src/arXiv_src_manifest.xml \
    --request-payer requester >/dev/null
echo "[bootstrap] AWS access OK"

if [[ "$AUTO_RUN" != "1" ]]; then
    echo
    echo "[bootstrap] AUTO_RUN=0 — skipping auto-launch. To start manually:"
    echo "  for n in \$(seq 0 $((SHARDS-1))); do"
    echo "    sudo -u $WORK_USER setsid nohup python3 $INSTALL_DIR/drive.py \\"
    echo "      --profile '' --staging-bucket $STAGING_BUCKET --select all \\"
    echo "      --shard \$n/$SHARDS -j \$(( \$(nproc) / $SHARDS )) \\"
    echo "      --workdir /tmp/arxiv_drive/shard\$n \\"
    echo "      </dev/null >$WORK_HOME/drive-\$n.log 2>&1 &"
    echo "  done"
    echo "  sudo -u $WORK_USER setsid nohup bash $WORK_HOME/watchdog.sh \\"
    echo "    </dev/null >>$WORK_HOME/watchdog.log 2>&1 &"
    exit 0
fi

# 8. Auto-launch SHARDS drive instances + the watchdog
CORES_PER_SHARD=$(( $(nproc) / SHARDS ))
if [[ "$CORES_PER_SHARD" -lt 1 ]]; then CORES_PER_SHARD=1; fi
echo "[bootstrap] launching $SHARDS drive shards × $CORES_PER_SHARD cores each"

mkdir -p /tmp/arxiv_drive
chown "$WORK_USER:$WORK_USER" /tmp/arxiv_drive

for n in $(seq 0 $((SHARDS - 1))); do
    workdir="/tmp/arxiv_drive/shard$n"
    mkdir -p "$workdir"
    chown "$WORK_USER:$WORK_USER" "$workdir"
    log="$WORK_HOME/drive-$n.log"
    sudo -u "$WORK_USER" bash -c "setsid nohup python3 $INSTALL_DIR/drive.py \
        --profile '' --staging-bucket $STAGING_BUCKET --select all \
        --shard $n/$SHARDS -j $CORES_PER_SHARD \
        --workdir $workdir \
        </dev/null >$log 2>&1 &"
done

echo "[bootstrap] launching watchdog"
sudo -u "$WORK_USER" bash -c "setsid nohup bash $WORK_HOME/watchdog.sh \
    </dev/null >>$WORK_HOME/watchdog.log 2>&1 &"

sleep 3
echo "[bootstrap] running processes:"
ps -ef | grep -E "drive.py|watchdog" | grep -v grep | awk '{print " ", $2, $8, $9, $10, $11, $12, $13}'
echo "[bootstrap] done."
