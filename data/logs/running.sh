#!/bin/bash

grep -rH "$(date +%F)" . | cut -d: -f1 | sort -u
