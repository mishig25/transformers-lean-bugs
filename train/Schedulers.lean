/-
  Exact (rational) models of the piecewise-linear LR lambdas in
  src/transformers/optimization.py, plus Float replays of the cosine ones.

  A value is a pair (num, den) with den > 0 meaning num/den; `eqQ` is cross-multiplication.
  Trainer usage: LambdaLR evaluates λ(0) at construction and λ(k) after the k-th
  scheduler.step(); optimizer step k (0-based) uses λ(k); λ(T) is set after the last step.
-/

def eqQ (a b : Int × Int) : Prop := a.1 * b.2 = b.1 * a.2
def leQ (a b : Int × Int) : Prop := a.1 * b.2 ≤ b.1 * a.2

/-! ## linear  (optimization.py:101-104) -/
def linear (W T s : Int) : Int × Int :=
  if s < W then (s, max 1 W) else (max 0 (T - s), max 1 (T - W))

theorem linear_peak_at_W (W T : Int) (h : W < T) : eqQ (linear W T W) (1, 1) := by
  unfold linear eqQ; simp; omega

theorem linear_end (W T : Int) (h : W ≤ T) : eqQ (linear W T T) (0, 1) := by
  unfold linear eqQ
  by_cases hTW : T < W
  · omega
  · simp [hTW]

theorem linear_warmup_monotone (W T s : Int) (h0 : 0 ≤ s) (h : s + 1 < W) :
    leQ (linear W T s) (linear W T (s + 1)) := by
  unfold linear leQ
  have h1 : s < W := by omega
  have h2 : s + 1 < W := h
  simp [h1, h2]
  have : max 1 W = W := by omega
  rw [this]; exact Int.mul_le_mul_of_nonneg_right (by omega) (by omega)

theorem linear_decay_monotone (W T s : Int) (hs : W ≤ s) :
    leQ (linear W T (s + 1)) (linear W T s) := by
  unfold linear leQ
  have h1 : ¬ s < W := by omega
  have h2 : ¬ s + 1 < W := by omega
  simp only [h1, h2, ite_false]
  exact Int.mul_le_mul_of_nonneg_right (by omega) (by omega)

theorem linear_nonneg (W T s : Int) (h0 : 0 ≤ s) : 0 ≤ (linear W T s).1 := by
  unfold linear; split <;> simp <;> omega

/-! ## constant_with_warmup (optimization.py:74-77) -/
def constWarm (W s : Int) : Int × Int := if s < W then (s, max 1 W) else (1, 1)
theorem constWarm_continuous (W : Int) (hW : 0 < W) :
    eqQ (constWarm W W) (1, 1) ∧ leQ (constWarm W (W - 1)) (1, 1) := by
  have h1 : W - 1 < W := by omega
  have h2 : ¬ W < W := Int.lt_irrefl W
  have hm : max 1 W = W := by omega
  unfold constWarm eqQ leQ; rw [if_neg h2, if_pos h1, hm]; simp; omega

/-! ## polynomial, power = 1 (optimization.py:221-239)
    λ(s) = ((li - le) * (1 - (s-W)/(T-W)) + le) / li  for W ≤ s ≤ T,  le/li for s > T.
    As a rational: ((li-le)*(T-s) + le*(T-W)) / ((T-W)*li).  Python divides by (T-W) = 0 when
    W = T, so we return `none` there. -/
def poly1 (W T li le s : Int) : Option (Int × Int) :=
  if s < W then some (s, max 1 W)
  else if s > T then some (le, li)
  else if T - W = 0 then none                               -- ZeroDivisionError in Python
  else some ((li - le) * (T - s) + le * (T - W), (T - W) * li)

theorem poly1_end (W T li le : Int) (h : W < T) : poly1 W T li le T = some (le * (T - W), (T - W) * li) := by
  unfold poly1
  have : ¬ T < W := by omega
  have h2 : ¬ T > T := by omega
  have h3 : ¬ T - W = 0 := by omega
  simp [this, h2, h3]

theorem poly1_end_value (W T li le : Int) : eqQ (le * (T - W), (T - W) * li) (le, li) := by
  unfold eqQ; simp only; grind

theorem poly1_peak (W T li le : Int) (h : W < T) (_hli : 0 < li) :
    ∃ v, poly1 W T li le W = some v ∧ eqQ v (1, 1) := by
  unfold poly1
  have : ¬ W < W := by omega
  have h2 : ¬ W > T := by omega
  have h3 : ¬ T - W = 0 := by omega
  simp only [this, h2, h3, ite_false]
  refine ⟨_, rfl, ?_⟩
  unfold eqQ; simp only
  grind

-- W = T: the final scheduler.step() (λ(T)) divides by zero.
example : poly1 5 5 1 0 5 = none := by decide

/-! ## WSD with linear warmup and linear decay (optimization.py:470-505) as rationals.
    m = min_lr_ratio = mn/md. -/
def wsdLin (W S D mn md s : Int) : Int × Int :=
  if s < W then
    -- progress*(1-m)+m = (s*(md-mn) + mn*max(1,W)) / (max(1,W)*md)
    (s * (md - mn) + mn * max 1 W, max 1 W * md)
  else if s < W + S then (1, 1)
  else if s < W + S + D then
    -- (1-progress)*(1-m)+m, progress = (s-W-S)/max(1,D)
    ((max 1 D - (s - W - S)) * (md - mn) + mn * max 1 D, max 1 D * md)
  else (mn, md)

theorem wsd_start (W S D mn md : Int) (hW : 0 < W) (_hmd : 0 < md) :
    eqQ (wsdLin W S D mn md 0) (mn, md) := by
  unfold wsdLin eqQ; simp [hW]
  have : max 1 W = W := by omega
  rw [this]; grind

theorem wsd_decay_start (W S D mn md : Int) (hS : 0 ≤ S) (hD : 0 < D) :
    eqQ (wsdLin W S D mn md (W + S)) (1, 1) := by
  unfold wsdLin eqQ
  have h1 : ¬ W + S < W := by omega
  have h2 : ¬ W + S < W + S := by omega
  have h3 : W + S < W + S + D := by omega
  simp only [h1, h2, h3, ite_false, ite_true]
  have : W + S - W - S = 0 := by omega
  rw [this, Int.sub_zero]
  have : max 1 D = D := by omega
  rw [this]
  grind

-- continuity at the end of decay: the last decay step's left-limit equals min_lr_ratio
theorem wsd_end (W S D mn md : Int) (hS : 0 ≤ S) (hD : 0 < D) :
    eqQ (wsdLin W S D mn md (W + S + D)) (mn, md) := by
  unfold wsdLin eqQ
  have h1 : ¬ W + S + D < W := by omega
  have h2 : ¬ W + S + D < W + S := by omega
  have h3 : ¬ W + S + D < W + S + D := by omega
  simp [h1, h2, h3]

-- Negative stable phase (num_training_steps < W + D): the lr jumps from ~1 down into the
-- middle of the decay when warmup ends.  Example W=10, D=10, T=15 -> S = -5.
#eval (List.range 16).map fun i => let v := wsdLin 10 (-5) 10 0 1 (Int.ofNat i); (i, v.1, v.2)

/-! ## cosine_warmup_with_min_lr  (optimization.py:389-411)
    warmup with warmup_lr_rate = r (rn/rd):  r + (1-r) * s / max(1, W-1)
    Documented intent: linear from r (at step 0) to the peak 1 (at the last warmup step W-1). -/
def cwmWarm (W rn rd s : Int) : Int × Int :=
  -- (rn * max(1,W-1) + (rd-rn) * s) / (rd * max(1, W-1))
  (rn * max 1 (W - 1) + (rd - rn) * s, rd * max 1 (W - 1))

theorem cwm_peak_at_last_warmup (W rn rd : Int) (hW : 2 ≤ W) :
    eqQ (cwmWarm W rn rd (W - 1)) (1, 1) := by
  unfold cwmWarm eqQ
  have : max 1 (W - 1) = W - 1 := by omega
  simp only [this, Int.mul_one, Int.one_mul]
  grind

-- Spec over all W ≥ 1 fails at W = 1: the single warmup step uses r, the peak is never used.
theorem cwm_peak_fails_W1 : ¬ ∀ W : Int, 1 ≤ W → eqQ (cwmWarm W 1 10 (W - 1)) (1, 1) := by
  intro h; have := h 1 (by decide); unfold eqQ cwmWarm at this; revert this; decide

/-! ## Float replays of the cosine lambdas, at the boundaries -/
def pi : Float := 3.141592653589793
def cosine (W T : Nat) (nc : Float) (s : Nat) : Float :=
  if s < W then s.toFloat / (max 1 W).toFloat
  else
    let p := (s - W).toFloat / (max 1 (T - W)).toFloat
    max 0 (0.5 * (1 + Float.cos (pi * nc * 2 * p)))

def cosineMin (W T : Nat) (nc mr : Float) (s : Nat) : Float :=
  if s < W then s.toFloat / (max 1 W).toFloat
  else
    let p := (s - W).toFloat / (max 1 (T - W)).toFloat
    let f := 0.5 * (1 + Float.cos (pi * nc * 2 * p))
    max 0 (f * (1 - mr) + mr)

def cosRestarts (W T : Nat) (nc : Nat) (s : Nat) : Float :=
  if s < W then s.toFloat / (max 1 W).toFloat
  else
    let p := (s - W).toFloat / (max 1 (T - W)).toFloat
    if p ≥ 1 then 0 else
    let x := nc.toFloat * p
    max 0 (0.5 * (1 + Float.cos (pi * (x - x.floor))))

def cwmRate (W T : Nat) (nc mr : Float) (r : Option Float) (s : Nat) : Float :=
  let sf := s.toFloat; let Wf := W.toFloat; let Tf := T.toFloat
  if sf < Wf then
    match r with
    | none => (sf + 1) / max 1 Wf
    | some r => r + (1 - r) * sf / (max 1 (Wf - 1))
  else
    let p := (sf - Wf + 1) / max 1 (Tf - Wf)
    let f := 0.5 * (1 + Float.cos (pi * nc * 2 * p))
    max 0 (f * (1 - mr) + mr)

-- (λ(W-1), λ(W), λ(T-1), λ(T)) for W=10, T=100
#eval (cosine 10 100 0.5 9, cosine 10 100 0.5 10, cosine 10 100 0.5 99, cosine 10 100 0.5 100)
#eval (cosineMin 10 100 0.5 0.1 9, cosineMin 10 100 0.5 0.1 10, cosineMin 10 100 0.5 0.1 100)
#eval (cosRestarts 10 100 2 10, cosRestarts 10 100 2 54, cosRestarts 10 100 2 55, cosRestarts 10 100 2 100)
-- cosine_warmup_with_min_lr: W=1 with warmup_lr_rate=0.1 -> step 0 uses 0.1, step 1 already decayed
#eval ((List.range 4).map fun s => cwmRate 1 100 0.5 0.0 (some 0.1) s)
#eval ((List.range 4).map fun s => cwmRate 2 100 0.5 0.0 (some 0.1) s)
#eval ((List.range 4).map fun s => cwmRate 1 100 0.5 0.0 none s)
-- cosine_warmup_with_min_lr: last used step T-1 reaches min_lr exactly (by design)
#eval (cwmRate 10 100 0.5 0.1 none 99, cwmRate 10 100 0.5 0.1 none 100)
