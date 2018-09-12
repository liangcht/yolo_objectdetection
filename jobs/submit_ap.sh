#!/bin/bash
OSName=$(uname -s)
cwd=$(dirname "${BASH_SOURCE[0]}")
job_json=${1:-$cwd/legacybash.json}
if [[ ! -f $job_json ]]; then
    echo "Cannot find the json file: $job_json"
    exit 1
fi
cmd=curl
if [[ $OSName == MINGW64* ]]; then
    # Windows needs winpty to get the password
    cmd="winpty curl"
fi
$cmd -k --ntlm --user REDMOND\\ehazar -X POST -d@$job_json -H "Content-Type: application/json" "https://philly/api/v2/submit"
