---- MODULE MCCache ----
EXTENDS Cache
Procs3 == {"p1","p2","p3"}
Commit3 == [p \in Procs3 |-> IF p = "p3" THEN "C2" ELSE "C1"]
Procs4 == {"p1","p2","p3","p4"}
Commit4 == [p \in Procs4 |-> IF p \in {"p3","p4"} THEN "C2" ELSE "C1"]
====
