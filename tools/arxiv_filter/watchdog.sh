#!/usr/bin/env bash
#
# Watchdog: polls staging marker count and terminates the instance when the
# full archive is complete.
#
# Run via: nohup bash watchdog.sh </dev/null >/home/ec2-user/watchdog.log 2>&1 &
#
# Termination policy:
#   - markers >= TARGET  → sync logs to S3, terminate the instance.
#   - markers <  TARGET  → keep polling, regardless of whether drive.py is alive.
#     (If drive.py crashed but the work isn't done, leaving the instance up
#     gives us a chance to investigate. The 30-day lifecycle on staging means
#     no compute cost is wasted forever; manual cleanup is the safety valve.)

set -u

REGION=us-east-1
BUCKET=christian-arxiv-staging
TARGET=${TARGET:-12374}
POLL_INTERVAL=${POLL_INTERVAL:-300}   # 5 min
LOG=/home/ec2-user/watchdog.log

INSTANCE_ID="$(
    TOKEN=$(curl -fs -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' \
            http://169.254.169.254/latest/api/token 2>/dev/null)
    if [[ -n "$TOKEN" ]]; then
        curl -fs -H "X-aws-ec2-metadata-token: $TOKEN" \
             http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null
    else
        curl -fs http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null
    fi
)"

if [[ -z "$INSTANCE_ID" ]]; then
    echo "[$(date -u +%FT%TZ)] [watchdog] could not get instance id; aborting" >&2
    exit 1
fi

echo "[$(date -u +%FT%TZ)] [watchdog] starting; instance=$INSTANCE_ID target=$TARGET poll=${POLL_INTERVAL}s"

while true; do
    count="$(aws s3 ls "s3://$BUCKET/markers/" 2>/dev/null | wc -l)"
    drive_alive="$(pgrep -fc "drive.py.*--select all" || echo 0)"
    echo "[$(date -u +%FT%TZ)] markers=$count drive_alive=$drive_alive"

    if [[ "$count" -ge "$TARGET" ]]; then
        echo "[$(date -u +%FT%TZ)] [watchdog] target reached. syncing logs + terminating."
        # Upload final logs before pulling the rug
        aws s3 cp /home/ec2-user/drive.log    "s3://$BUCKET/logs/drive-final.log"    || true
        aws s3 cp "$LOG"                       "s3://$BUCKET/logs/watchdog-final.log" || true
        # 30s grace so the log uploads above show up if you're tailing
        sleep 30
        aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID"
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done
