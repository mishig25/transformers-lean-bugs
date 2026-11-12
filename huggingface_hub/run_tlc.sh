#!/bin/bash
# Translate the PlusCal specs and model-check every configuration; logs go to results/<cfg>.log.
# -deadlock is used only where idle worker threads legitimately wait forever (CommitScheduler workers).
cd "$(dirname "$0")"
JAR=${TLA2TOOLS:-tla2tools.jar}
TLC="java -XX:+UseParallelGC -Xmx1500m -cp $JAR tlc2.TLC -workers 1"
for s in LocalDir Cache UploadPipeline CommitScheduler; do java -cp $JAR pcal.trans $s.tla >/dev/null 2>&1; rm -f $s.cfg $s.old; done
run() { # module cfg [extra flags]
  $TLC $3 -config $2.cfg $1.tla > results/$2.log 2>&1
  printf "%-22s %-18s %s | %s\n" "$2" "$1" "$(grep -E -m1 'violated|Deadlock reached|No error has been found|Temporal properties were violated' results/$2.log)" "$(grep -E 'states generated, [0-9,]+ distinct' results/$2.log | tail -1)"
}
for c in ld_same ld_same_lfs ld_copy ld_diff ld_fixed_same ld_fixed_same_lfs ld_fixed_copy ld_fixed_diff ld_fixed_diff2; do run MCLocalDir $c; done
for c in cache_sym cache_sym_shared cache_nosym cache_nosym_ret; do run MCCache $c; done
for c in pipe_bug pipe_bug_rest pipe_nofail pipe_fixed; do run UploadPipeline $c; done
for c in cs_fine cs_multigrain cs_coarse cs_lazyrace; do run CommitScheduler $c -deadlock; done
rm -f *_TTrace_*; rm -rf states
