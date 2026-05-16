# Stage 3 Runbook — Full Arxiv Archive Filter on EC2

End-to-end procedure for running the full arxiv source filter against all
12,374 tarballs (3M papers) on a single EC2 spot instance in `us-east-1`.

**Estimated cost:** ~$30 total
- EC2 c6i.16xlarge spot, ~35 h × $0.50/h = ~$18
- Source bucket reads (intra-region, free) = $0
- S3 GET request charges (12k requests) ≈ $0.01
- Output egress staging → ceph (~88 GB × $0.09) ≈ $8
- Staging S3 storage (~1 week before sync to ceph) ≈ $0.50

**Estimated wall-clock:** ~35 hours from launch to filtered-output complete.

## Pre-flight checklist

Already done in earlier setup (2026-05-15):

- [x] AWS account `015647311828`, IAM user `arxiv-bulk` with read on `arxiv` + read/write on `christian-arxiv-staging`.
- [x] Staging bucket `christian-arxiv-staging` with 30-day lifecycle, public access blocked.
- [x] IAM role `arxiv-filter-ec2` + instance profile of the same name. Inline policies: `s3-arxiv-and-staging`, `terminate-self`. Managed: `AmazonSSMManagedInstanceCore`.
- [x] EC2 spot quota raised to 128 vCPU (request ID b83173d6af394f50b55c27a8a07a3dd89HzxZMDw).
- [x] `filter.py`, `drive.py`, `bootstrap.sh`, `watchdog.sh` uploaded to `s3://christian-arxiv-staging/scripts/`.

Verify by listing scripts:
```bash
aws --profile arxiv s3 ls s3://christian-arxiv-staging/scripts/
```

If scripts are stale (you edited them locally), re-upload:
```bash
aws --profile arxiv s3 cp tools/arxiv_filter/filter.py    s3://christian-arxiv-staging/scripts/filter.py
aws --profile arxiv s3 cp tools/arxiv_filter/drive.py     s3://christian-arxiv-staging/scripts/drive.py
aws --profile arxiv s3 cp tools/arxiv_filter/bootstrap.sh s3://christian-arxiv-staging/scripts/bootstrap.sh
aws --profile arxiv s3 cp tools/arxiv_filter/watchdog.sh  s3://christian-arxiv-staging/scripts/watchdog.sh
```

## Step 1: Launch the EC2 instance

Either via console or via CLI. Both produce the same instance shape.

### Option A: Console

1. EC2 → Launch instances. **Region: us-east-1 (N. Virginia).**
2. Name: `arxiv-filter`.
3. AMI: **Amazon Linux 2023** (latest, x86_64).
4. Instance type: **c6i.16xlarge** (64 vCPU, 128 GB RAM).
5. Key pair: none needed (we'll connect via SSM Session Manager). Choose "Proceed without a key pair" if prompted.
6. Network settings: defaults (default VPC, default subnet, "no public IP" is fine since SSM works through the EC2 endpoint).
7. Storage: **50 GiB gp3** root volume.
8. Advanced → IAM instance profile: **`arxiv-filter-ec2`**.
9. Advanced → Purchasing option: check **Request Spot instances**. Persist after interruption: yes. Interruption behavior: **terminate** (we don't care about preserving the instance — work resumes from `s3://staging/markers/`).
10. Advanced → User data: paste this one-liner to run bootstrap automatically on first boot:
    ```bash
    #!/bin/bash
    aws s3 cp s3://christian-arxiv-staging/scripts/bootstrap.sh /tmp/bootstrap.sh && bash /tmp/bootstrap.sh
    ```
11. Launch.

### Option B: CLI (single command)

From your dev machine:

```bash
AMI=$(aws ssm get-parameters \
  --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query 'Parameters[0].Value' --output text)

aws ec2 run-instances \
  --image-id "$AMI" \
  --instance-type c6i.16xlarge \
  --iam-instance-profile Name=arxiv-filter-ec2 \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=50,VolumeType=gp3}' \
  --instance-market-options 'MarketType=spot' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=arxiv-filter}]' \
  --user-data '#!/bin/bash
aws s3 cp s3://christian-arxiv-staging/scripts/bootstrap.sh /tmp/bootstrap.sh && bash /tmp/bootstrap.sh' \
  --query 'Instances[0].InstanceId' --output text
```

Save the returned instance ID — you'll need it for SSM.

## Step 2: Wait for bootstrap + connect

Bootstrap usually takes 2–3 minutes (mostly `dnf install` and pandoc download).
Watch for SSM availability:

```bash
INSTANCE_ID="i-XXXXX"  # from step 1
aws ssm describe-instance-information \
  --filters Key=InstanceIds,Values=$INSTANCE_ID \
  --query 'InstanceInformationList[0].PingStatus' --output text
# Wait until this prints "Online"
```

Then connect:

```bash
aws ssm start-session --target $INSTANCE_ID
```

Once in the shell:

```bash
# Confirm bootstrap ran cleanly (look for the "[bootstrap] done" line)
sudo tail -50 /var/log/cloud-init-output.log

# Verify tools
pandoc --version | head -1
python3 -c "import zstandard; print(zstandard.__version__)"
aws sts get-caller-identity
ls ~/arxiv_filter/
```

If anything is missing, re-run the bootstrap manually:
```bash
sudo bash /tmp/bootstrap.sh
```

## Step 3: Kick off the full filter run

```bash
cd ~/arxiv_filter
# --profile '' tells drive.py to use the instance role instead of a named profile
setsid nohup python3 drive.py \
  --profile '' \
  --staging-bucket christian-arxiv-staging \
  --select all \
  --workdir /tmp/arxiv_drive \
  </dev/null > /home/ec2-user/drive.log 2>&1 &
tail -f /home/ec2-user/drive.log
```

`-j` defaults to all cores (64 on c6i.16xlarge). No need to override.

(`setsid` + `</dev/null` are important when launching from SSM Send-Command —
otherwise the SSM session keeps the process tied to itself and it dies when SSM
returns.)

## Step 4: Start the auto-shutdown watchdog

Polls the staging marker count every 5 minutes; when it reaches 12,374 the
watchdog syncs logs to S3 and calls `ec2:TerminateInstances` on itself.
Required IAM permission is already on the role (`terminate-self` inline
policy).

```bash
aws s3 cp s3://christian-arxiv-staging/scripts/watchdog.sh /home/ec2-user/watchdog.sh
chmod +x /home/ec2-user/watchdog.sh
setsid nohup bash /home/ec2-user/watchdog.sh \
  </dev/null >>/home/ec2-user/watchdog.log 2>&1 &
```

Verify:
```bash
tail -5 /home/ec2-user/watchdog.log
# Should show: "[watchdog] starting; instance=i-... target=12374 poll=300s"
# Then a periodic "markers=N drive_alive=1" line every 5 min.
```

If drive.py crashes before reaching the target, the watchdog stays alive but
will *not* terminate — the instance is preserved for investigation. Manual
intervention (re-launch drive.py, or terminate) is required in that case.

## Step 5: Monitor progress

From the EC2 shell:
```bash
tail -f ~/arxiv_filter/drive.log
```

From your dev machine, periodically check how many markers exist:
```bash
aws --profile arxiv s3 ls s3://christian-arxiv-staging/markers/ | wc -l
# 12374 means done
```

Expected pace once warmed up: roughly **350 markers/hour** at c6i.16xlarge speed
(20 papers/s × ~140 papers/tarball ≈ 12 s/tarball average). The first tarballs
of the manifest are smaller (1990s papers) and will fly; recent ones are larger
and slower.

## Step 6: Handle spot interruption

If AWS reclaims the spot instance mid-run:

- Output already in `s3://staging/filtered/` is durable. Markers in
  `s3://staging/markers/` mark which tarballs are done.
- Re-launch the instance (same procedure). `drive.py` will skip any tarball
  whose marker is present and pick up from the next unprocessed one.
- Watchdog must be restarted too (it's not auto-launched by bootstrap). Easy
  to forget — repeat step 4 on the new instance.
- The work-in-flight tarball is lost (small — one tarball of work, ~1 minute).

For resilience without intervention, you can rely on the spot-request "persist
after interruption" setting from step 1; AWS will provision a new instance when
spot capacity returns. Or just check on it periodically and re-launch manually.

## Step 7: Sync output to ceph

Once `markers/` contains all 12,374 markers, pull the filtered output to your
ceph storage:

```bash
# From the machine that mounts ceph (NOT the EC2 instance)
aws --profile arxiv s3 sync \
  s3://christian-arxiv-staging/filtered/ \
  /ceph/path/arxiv/base/20260515/
```

Expected: ~88 GB across 12,374 `.tar.zst` files. Sync speed depends on your
home/ceph bandwidth — at 100 Mbps that's ~2 hours; at 1 Gbps ~12 min.

Verify checksums against the source manifest (which has md5 per source
tarball — note these are md5 of the *input* tarballs, not our filtered output,
so we'd want to compute our own checksums separately if integrity is critical).

## Step 8: Tear down

If the watchdog ran to completion, the instance is already terminated — skip
to staging cleanup below. Otherwise:

```bash
# Terminate the EC2 instance (saves spot $)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

Once output is on ceph:

```bash
# Clear staging bucket (lifecycle would do this in 30 days; sooner is cheaper)
aws s3 rm s3://christian-arxiv-staging/filtered/ --recursive
aws s3 rm s3://christian-arxiv-staging/markers/  --recursive
aws s3 rm s3://christian-arxiv-staging/logs/     --recursive
# Leave scripts/ in place for future re-runs

# Optionally rotate the arxiv-bulk IAM access key now that bulk pull is done
```

## Troubleshooting

**Bootstrap failed with "Unable to locate credentials" or 403**
The instance profile didn't attach. Check:
```bash
aws sts get-caller-identity
# Should show arn:aws:sts::015647311828:assumed-role/arxiv-filter-ec2/...
```
If it shows nothing, the profile is missing — terminate and re-launch with
`--iam-instance-profile Name=arxiv-filter-ec2`.

**Pandoc segfaults or returns immediately**
Confirm version:
```bash
pandoc --version | head -1
# Expected: pandoc 3.6
```
If older (2.x), the bootstrap fallback installed the system pandoc instead of
the binary. Re-run bootstrap, or install manually:
```bash
curl -fL https://github.com/jgm/pandoc/releases/download/3.6/pandoc-3.6-linux-amd64.tar.gz \
  | sudo tar xzf - -C /usr/local --strip-components=1
```

**Filter throughput unexpectedly low (< 10 papers/s)**
Confirm 64 cores are being used:
```bash
ps -ef | grep filter.py | wc -l   # should be ~64 + 1
nproc                              # confirm 64
```
If `nproc` shows fewer, instance type may have downgraded. Stop and re-launch
on the right type.

**Disk full**
`/tmp` is the default scratch path. Each tarball needs ~1 GB scratch; the
driver cleans up between tarballs. If filling up, check whether `/tmp` is
backed by tmpfs (RAM) with too small a size — for AL2023 it's 50% of RAM
which is fine. If on a small system, override:
```bash
python3 drive.py ... --workdir /var/scratch/arxiv_drive
```
(after creating the dir on the larger volume).

## Cost monitoring

Set a budget alarm to catch any runaway:

```bash
aws budgets create-budget --account-id 015647311828 --budget '{
  "BudgetName": "arxiv-filter-run",
  "BudgetLimit": {"Amount": "75", "Unit": "USD"},
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}'
```

(Optional — the run is bounded by the work itself; the spot instance
auto-terminates when drive.py exits if you start it via `bash`. Keep an eye on
it.)
