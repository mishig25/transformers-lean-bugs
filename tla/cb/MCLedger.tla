---- MODULE MCLedger ----
EXTENDS CBLedger
CONSTANTS a, b, c
LReq == {a, b, c}
LCont == (a :> <<"x", "y", "z">>) @@ (b :> <<"x", "y", "w">>) @@ (c :> <<"x", "y", "z">>)
====
