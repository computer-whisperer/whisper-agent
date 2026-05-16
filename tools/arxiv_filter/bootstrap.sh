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

# 6. Pull filter.py + drive.py from staging
mkdir -p "$INSTALL_DIR"
aws s3 cp "s3://$STAGING_BUCKET/scripts/filter.py" "$INSTALL_DIR/filter.py"
aws s3 cp "s3://$STAGING_BUCKET/scripts/drive.py"  "$INSTALL_DIR/drive.py"
chown -R "$WORK_USER:$WORK_USER" "$INSTALL_DIR"
chmod +x "$INSTALL_DIR"/*.py
echo "[bootstrap] installed scripts: $(ls "$INSTALL_DIR")"

# 7. Verify AWS access via instance metadata
echo "[bootstrap] verifying AWS access via instance profile..."
aws sts get-caller-identity --query 'Arn' --output text
aws s3 ls "s3://$STAGING_BUCKET/" >/dev/null
aws s3api head-object --bucket arxiv --key src/arXiv_src_manifest.xml \
    --request-payer requester >/dev/null
echo "[bootstrap] AWS access OK"

echo
echo "[bootstrap] done. Next step (as $WORK_USER):"
echo "  cd $INSTALL_DIR"
echo "  python3 drive.py --profile '' --staging-bucket $STAGING_BUCKET --select all"
echo
echo "Suggested wrapper for resumability + logging:"
echo "  nohup python3 drive.py --profile '' --select all >drive.log 2>&1 &"
echo "  tail -f drive.log"
