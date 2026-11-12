---- MODULE MC ----
EXTENDS CBLoop
CONSTANTS r1, r2, r3
\* two requests, short prompts, 3 blocks of 2 tokens: two concurrent requests cannot both finish in cache
Req2 == {r1, r2}
P2 == [r \in Req2 |-> 2]
M2 == [r \in Req2 |-> 3]
Req3 == {r1, r2, r3}
P3 == [r \in Req3 |-> 2]
M3 == [r \in Req3 |-> 3]
P3b == (r1 :> 2) @@ (r2 :> 3) @@ (r3 :> 1)
M3b == (r1 :> 3) @@ (r2 :> 2) @@ (r3 :> 3)
\* r1 alone needs more KV than the cache holds (2 + 6 - 1 = 7 tokens > 3 blocks * 2)
PBig == [r \in Req2 |-> 2]
MBig == (r1 :> 6) @@ (r2 :> 2)
====
