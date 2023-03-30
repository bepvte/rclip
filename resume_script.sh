#!/usr/bin/env bash

if [[ $CRTOOLS_SCRIPT_ACTION == "post-resume" ]]; then
	pkill -SIGCONT rclip
fi
