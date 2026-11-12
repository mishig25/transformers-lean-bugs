---- MODULE MCOut ----
EXTENDS CBOutput
CONSTANTS a, b, s
ReqServe == {s}
KServe == [r \in {s} |-> 2]
ReqDoc == {a, b}
KDoc == [r \in {a, b} |-> 1]
ReqRid == {a, b}
KRid == [r \in {a, b} |-> 2]
KRid3 == [r \in {a, b} |-> 3]
====
