export MODEL_DIR=${PHILLY_VC_HDFS_DIRECTORY:-/hdfs/input}/sys/jobs/${PHILLY_JOB_ID}/models
export LOG_DIR=$HOME/logs/$PHILLY_ATTEMPT_ID
export STDOUT_DIR=${LOG_DIR/logs/stdout}
export DATA_DIR=/${PHILLY_VC_HDFS_DIRECTORY:-/hdfs/input}/data
