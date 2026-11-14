#!/bin/bash
# Stage the round-2 clone's HEAD as code/r2_<sha8>.tar.gz and refresh SHA256SUMS; prints the path.
set -e
S=/ceph/srv/dwong/r2; cd /work/dwong/r2_repo
SHA=$(git rev-parse --short=8 HEAD); f=code/r2_$SHA.tar.gz
[ -f $S/$f ] || git archive --format=tar.gz HEAD > $S/$f
cd $S && find . -type f ! -name SHA256SUMS ! -name ".*" | sed "s#^\./##" | sort | xargs sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS
echo $f
