#!/bin/bash
# Re-runs every TLC configuration and prints a one-line verdict per config.
cd "$(dirname "$0")"
TLC="java -XX:+UseParallelGC -cp ${TLA2TOOLS:-tla2tools.jar} tlc2.TLC -workers 4 -deadlock"
run() { # spec cfg
  out="results/${2%.cfg}.txt"; mkdir -p results
  $TLC -config "$2" "$1" > "$out" 2>&1
  verdict=$(grep -m1 -o "Invariant [A-Za-z]* is violated\|Temporal propert[a-z]* [A-Za-z ,]* violated\|No error has been found" "$out")
  states=$(grep -m1 -o "[0-9,]* distinct states found, 0 states left" "$out" | cut -d' ' -f1)
  crash=$(grep -m1 -o 'crashed |-> "[A-Za-z]*"' "$out" | grep -v '"none"' | tail -1)
  printf "%-28s %-60s %s %s\n" "$2" "$verdict" "${states:+states=$states}" "$crash"
}
for c in sync_cancel sync_cancel_atomic sync_cancel_atomic_fixed async_share_cancel async_share_cancel_fixed \
         async_noshare_cancel sync_hardstop sync_oversize big_sync_nocancel big_async_nocancel \
         big_sync_fixed big_async_fixed all_fixed_sync all_fixed_async; do run MC.tla $c.cfg; done
for c in out_serve out_serve_fixed out_doc out_rid out_rid3; do run MCOut.tla $c.cfg; done
run MCLedger.tla ledger.cfg
