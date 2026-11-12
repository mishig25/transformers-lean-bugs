---- MODULE MCLocalDir ----
EXTENDS LocalDir
WantSame2 == [p \in {"p1","p2"} |-> 1]
WantSame3 == [p \in {"p1","p2","p3"} |-> 1]
WantDiff  == [p \in {"p1","p2"} |-> IF p = "p1" THEN 1 ELSE 2]
WantDiff3 == [p \in {"p1","p2","p3"} |-> IF p = "p1" THEN 1 ELSE 2]
====
