#!/usr/bin/env bash
# 📡 Route B (DECISION_LOG 597) — create / update the recorder's AWS resources from the operator's Mac (aws CLI, admin profile).
# Idempotent: re-running updates the bucket settings, the policies, the code in S3 and the systemd unit on the existing server (via SSM)
# and restarts the service. Nothing the bot runs on is created, rebuilt or replaced — the ONLY bot-side change is an inline policy on its
# instance role allowing PutObject to s3://$BUCKET/bot/* (the Frenzy list). NOTE: that role is shared by every EB environment of the account.
#   bucket   s3://$BUCKET — private (public access blocked, HTTPS-only policy), SSE-S3, data/ → STANDARD_IA after 60 days,
#            incomplete multipart uploads aborted after 7 days. No expiry (the research data is kept; decide a retention later).
#   role     scalpars-recorder-role (+ instance profile): SSM managed core · Put data/* + health/* + features/* · Get code/* + bot/* (it cannot rewrite
#            its own code)
#   network  security group scalpars-recorder-sg in the default VPC: NO inbound rules (managed through SSM, no SSH keys)
#   server   t4g.small, Amazon Linux 2023 arm64, 30 GB gp3 encrypted, IMDSv2 only, tag Name=scalpars-recorder,
#            systemd service scalpars-recorder (Restart=always, SIGTERM → clean file close; pulls code/recorder.py from S3 on every start)
# Usage: scripts/route_b/deploy_aws.sh            (create or update everything)
#        scripts/route_b/deploy_aws.sh --code     (upload the code + restart the service only)
set -euo pipefail
REGION=eu-west-1
BUCKET=scalpars-route-b-recorder-euw1
ROLE=scalpars-recorder-role
SG=scalpars-recorder-sg
NAME=scalpars-recorder
BOT_ROLE=aws-elasticbeanstalk-ec2-role
WEBSOCKETS=13.1
MAX_HOT=120
HERE="$(cd "$(dirname "$0")" && pwd)"
AWS="aws --region $REGION"

instance_id() {   # the first RUNNING / pending instance with the tag (empty if none)
  $AWS ec2 describe-instances --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=pending,running" \
    --query 'Reservations[].Instances[].InstanceId' --output text | awk '{print $1}'
}

unit_text() {
  cat <<UNIT
[Unit]
Description=Route B recorder (DECISION_LOG 597)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0
[Service]
Environment=BUCKET=$BUCKET
Environment=DATA_DIR=/var/lib/scalpars-recorder
Environment=MAX_HOT=$MAX_HOT
Environment=AWS_DEFAULT_REGION=$REGION
ExecStartPre=-/usr/bin/aws s3 cp --only-show-errors s3://$BUCKET/code/recorder.py /opt/scalpars-recorder/recorder.py
ExecStartPre=-/usr/bin/aws s3 cp --only-show-errors s3://$BUCKET/code/features.py /opt/scalpars-recorder/features.py
ExecStart=/usr/bin/python3.11 /opt/scalpars-recorder/recorder.py
KillSignal=SIGTERM
TimeoutStopSec=30
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
UNIT
}

upload_code() {   # the recorder + the per-minute summary builder it runs every hour
  $AWS s3 cp --only-show-errors "$HERE/recorder.py" "s3://$BUCKET/code/recorder.py"
  $AWS s3 cp --only-show-errors "$HERE/features.py" "s3://$BUCKET/code/features.py"
}

ssm_run() {   # instance, commands (JSON array) → prints the command id; fails loudly
  $AWS ssm send-command --instance-ids "$1" --document-name AWS-RunShellScript --parameters "commands=$2" --query 'Command.CommandId' --output text
}

update_unit_and_restart() {
  local iid="$1" b64
  b64="$(unit_text | base64 | tr -d '\n')"
  ssm_run "$iid" "[\"echo $b64 | base64 -d > /etc/systemd/system/scalpars-recorder.service\",\"systemctl daemon-reload\",\"systemctl restart scalpars-recorder\",\"sleep 5\",\"systemctl is-active scalpars-recorder\"]"
}

if [[ "${1:-}" == "--code" ]]; then
  upload_code                                                                             # a failed upload stops here (set -e)
  iid="$(instance_id)"; [[ -n "$iid" ]] || { echo "no running $NAME instance"; exit 1; }
  echo "code uploaded; unit refreshed + restart command $(update_unit_and_restart "$iid") sent to $iid (IAM changes need the full run)"; exit 0
fi

# ── bucket ──
if ! $AWS s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  $AWS s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint=$REGION >/dev/null
  echo "bucket created: $BUCKET"
fi
$AWS s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
$AWS s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
$AWS s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration \
  '{"Rules":[{"ID":"data-to-ia-60d","Status":"Enabled","Filter":{"Prefix":"data/"},"Transitions":[{"Days":60,"StorageClass":"STANDARD_IA"}]},
             {"ID":"abort-mpu-7d","Status":"Enabled","Filter":{"Prefix":""},"AbortIncompleteMultipartUpload":{"DaysAfterInitiation":7}}]}' >/dev/null
$AWS s3api put-bucket-policy --bucket "$BUCKET" --policy "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Sid\":\"HttpsOnly\",\"Effect\":\"Deny\",
  \"Principal\":\"*\",\"Action\":\"s3:*\",\"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"],
  \"Condition\":{\"Bool\":{\"aws:SecureTransport\":\"false\"}}}]}"
upload_code

# ── IAM: the recorder's role ──
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  echo "role created: $ROLE"
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name route-b-bucket --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[
  {\"Effect\":\"Allow\",\"Action\":[\"s3:PutObject\"],\"Resource\":[\"arn:aws:s3:::$BUCKET/data/*\",\"arn:aws:s3:::$BUCKET/health/*\",\"arn:aws:s3:::$BUCKET/features/*\"]},
  {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\"],\"Resource\":[\"arn:aws:s3:::$BUCKET/code/*\",\"arn:aws:s3:::$BUCKET/bot/*\"]}]}"
if ! aws iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
  echo "instance profile created; waiting for IAM to propagate"; sleep 15
fi
# the bot may write ONLY its Frenzy list into the bucket
aws iam put-role-policy --role-name "$BOT_ROLE" --policy-name route-b-bot-list --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[
  {\"Effect\":\"Allow\",\"Action\":[\"s3:PutObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/bot/*\"}]}"

# ── network: no inbound ──
VPC="$($AWS ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
SGID="$($AWS ec2 describe-security-groups --filters "Name=group-name,Values=$SG" "Name=vpc-id,Values=$VPC" --query 'SecurityGroups[0].GroupId' --output text)"
if [[ "$SGID" == "None" || -z "$SGID" ]]; then
  SGID="$($AWS ec2 create-security-group --group-name "$SG" --description "Route B recorder - no inbound" --vpc-id "$VPC" --query GroupId --output text)"
  echo "security group created: $SGID"
fi

# ── server ──
IID="$(instance_id)"
if [[ -z "$IID" ]]; then
  if [[ -n "$($AWS ec2 describe-instances --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=stopping,stopped" \
        --query 'Reservations[].Instances[].InstanceId' --output text)" ]]; then
    echo "a STOPPED $NAME instance exists — start it (aws ec2 start-instances) instead of launching a second one"; exit 1
  fi
  AMI="$($AWS ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 --query Parameter.Value --output text)"
  UD="$(mktemp)"
  {
    echo '#!/bin/bash'
    echo 'set -e'
    echo 'dnf install -y python3.11 python3.11-pip'
    echo "python3.11 -m pip install 'websockets==$WEBSOCKETS'"
    echo 'mkdir -p /opt/scalpars-recorder /var/lib/scalpars-recorder'
    echo "sed -i 's/^#\\?SystemMaxUse=.*/SystemMaxUse=500M/' /etc/systemd/journald.conf && systemctl restart systemd-journald"
    echo "echo $(unit_text | base64 | tr -d '\n') | base64 -d > /etc/systemd/system/scalpars-recorder.service"
    echo 'systemctl daemon-reload'
    echo 'systemctl enable --now scalpars-recorder'
  } > "$UD"
  IID="$($AWS ec2 run-instances --image-id "$AMI" --instance-type t4g.small --security-group-ids "$SGID" \
    --iam-instance-profile Name="$ROLE" --metadata-options HttpTokens=required,HttpEndpoint=enabled \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3","Encrypted":true,"DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=scalpars-route-b}]" \
    --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text)"
  rm -f "$UD"
  echo "instance launched: $IID"
else
  echo "instance exists: $IID — unit + code updated; command $(update_unit_and_restart "$IID")"
fi
