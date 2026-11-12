#!/bin/sh
# Re-run every TLC check. Note: pcal.trans writes <Module>.cfg; translations are already in the .tla files.
cd "$(dirname "$0")"
TLC="java -XX:+UseParallelGC -Xmx1500m -cp ${TLA2TOOLS:-tla2tools.jar} tlc2.TLC -workers 1"
run() { spec=$1; cfg=$2; flags=$3
  r=$($TLC $flags $spec.tla -config $cfg 2>&1 | grep -E "is violated|states generated,|No error has been found" | tr '\n' ' ')
  printf '%-28s %s\n' "$cfg" "$r"; }
run ShuffleResume ShuffleResume_arrow.cfg -deadlock
run ShuffleResume ShuffleResume_python.cfg -deadlock
run ShuffleResume ShuffleResume_fixed.cfg -deadlock
run StepResume StepResume_arrow.cfg -deadlock
run StepResume StepResume_tablefix.cfg -deadlock
run StepResume StepResume_python.cfg -deadlock
run LoaderResume LoaderResume_crash.cfg -deadlock
run LoaderResume LoaderResume_sticky.cfg -deadlock
run LoaderResume LoaderResume_report.cfg -deadlock
run LoaderResume LoaderResume_fixed.cfg -deadlock
run MapCache MapCache.cfg -deadlock
run MapCache MapCache_droplast.cfg -deadlock
run Extract Extract.cfg ""
run Extract Extract_fixed.cfg ""
run DistShards DistShards.cfg -deadlock
run DistShards DistShards_unseeded.cfg -deadlock
rm -f *_TTrace_*.tla *_TTrace_*.bin; rm -rf states
