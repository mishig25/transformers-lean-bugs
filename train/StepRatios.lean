/-
  Model of TrainerState.compute_steps (src/transformers/trainer_callback.py:156)
  and of the `load_best_model_at_end` validation in TrainingArguments.__post_init__
  (src/transformers/training_args.py:1755-1768).

  Python:
      for kind in ("logging", "eval", "save"):
          n = getattr(args, f"{kind}_steps")
          if n < 1: n = math.ceil(max_steps * n)          # ratio -> absolute
  and, at argument-parsing time (before max_steps is known):
      if eval_steps < 1 and save_steps < 1:
          require (save_steps*1e6) % (eval_steps*1e6) == 0  # "save is a multiple of eval"

  We model a ratio as p/q with exact rationals (so floating point does not
  even enter): absolute steps = ceil(M * p / q) = (M*p + q - 1) / q.
-/

def ceilDiv (a b : Nat) : Nat := (a + b - 1) / b

/-- absolute steps for ratio p/q of max_steps M (exact arithmetic) -/
def absSteps (M p q : Nat) : Nat := ceilDiv (M * p) q

/-- The property `load_best_model_at_end` relies on: every save step is an eval step,
    i.e. state.save_steps % state.eval_steps == 0. The TrainingArguments check only
    establishes `save_ratio = k * eval_ratio`. -/
def SpecSaveIsMultipleOfEval : Prop :=
  ∀ M p q k : Nat, 0 < M → 0 < p → 0 < k → k * p < q →
    absSteps M (k * p) q % absSteps M p q = 0

-- Counterexample search: eval_steps=0.1, save_steps=0.2 (p=1,q=10,k=2) for max_steps 1..60
#eval (List.range 60).filterMap fun i =>
  let M := i + 1
  let e := absSteps M 1 10
  let s := absSteps M 2 10
  if s % e != 0 then some (M, e, s) else none

-- eval_steps=0.05, save_steps=0.1
#eval (List.range 60).filterMap fun i =>
  let M := i + 1
  let e := absSteps M 1 20
  let s := absSteps M 2 20
  if s % e != 0 then some (M, e, s) else none

/-- The spec is FALSE: max_steps = 25, eval_steps = 0.1, save_steps = 0.2 gives
    eval every 3 steps and save every 5 steps. -/
theorem spec_save_multiple_of_eval_fails : ¬ SpecSaveIsMultipleOfEval := by
  intro h
  have := h 25 1 10 2 (by decide) (by decide) (by decide) (by decide)
  revert this
  decide

/-- concrete instance, as the Trainer computes it -/
example : absSteps 25 1 10 = 3 ∧ absSteps 25 2 10 = 5 := by decide

/-- What *does* hold: when the ratio divides max_steps exactly, the multiple is preserved. -/
theorem save_multiple_when_exact (M p q k c : Nat) (hq : 0 < q) (hc : M * p = q * c) :
    absSteps M (k * p) q = k * absSteps M p q := by
  unfold absSteps ceilDiv
  have h1 : M * (k * p) = k * (M * p) := by
    rw [← Nat.mul_assoc, Nat.mul_comm M k, Nat.mul_assoc]
  rw [h1, hc]
  have e1 : k * (q * c) + q - 1 = (q - 1) + q * (k * c) := by
    have : k * (q * c) = q * (k * c) := by
      rw [← Nat.mul_assoc, Nat.mul_comm k q, Nat.mul_assoc]
    omega
  have e2 : q * c + q - 1 = (q - 1) + q * c := by omega
  rw [e1, e2, Nat.add_mul_div_left _ _ hq, Nat.add_mul_div_left _ _ hq]
  have : (q - 1) / q = 0 := Nat.div_eq_of_lt (by omega)
  rw [this]; simp

/-- A fix that does hold for all M: derive save from the *absolute* eval steps,
    save_abs := k * eval_abs.  Trivially a multiple. -/
theorem fixed_save_multiple (M p q k : Nat) :
    (k * absSteps M p q) % absSteps M p q = 0 := by
  exact Nat.mul_mod_left _ _

/-
  Floating point: Python computes math.ceil(max_steps * ratio) in IEEE doubles.
  Lean's Float is IEEE binary64, so we can replay the exact Python expression.
-/
-- For the common ratios 0.01..0.9 that we tried, M*r rounds to the exact ceil for M <= 2000
-- (no mismatch).  0.07 is an exception: 100*0.07 = 7.000000000000001 -> ceil 8.
#eval ((List.range 2000).filterMap fun i =>
  let M := i + 1
  let f := (Float.ceil (M.toFloat * 0.1)).toUInt64.toNat
  if f != absSteps M 1 10 then some M else none).take 30
#eval ((List.range 2000).filterMap fun i =>
  let M := i + 1
  let f := (Float.ceil (M.toFloat * 0.07)).toUInt64.toNat
  if f != absSteps M 7 100 then some (M, f, absSteps M 7 100) else none).take 8
