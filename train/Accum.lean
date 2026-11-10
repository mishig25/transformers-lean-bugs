/-
  Model of Trainer._run_epoch / set_initial_training_values / _init_training_state
  (src/transformers/trainer.py:1678-1713, 1819-1966, 2487-2555).

  Per epoch (G = gradient_accumulation_steps, S = steps_in_epoch, U = num_update_steps_per_epoch):
      remainder = S % G or G
      for update_step in range(num_update_steps_trained, U):
          num_batches = G if update_step != U-1 else remainder
          batch_samples = next(it) up to num_batches times (stops at StopIteration)
          for inputs in batch_samples:
              step += 1
              do_sync_step = (step+1) % G == 0 or (step+1) == S
              ... if do_sync_step: optimizer.step(); zero_grad(); global_step += 1
  Map-style dataloader:  S = len(dl) = L, U = max(ceil(L/G), 1), the iterator yields L batches.
  Iterable (no __len__): S = max_steps * G, U = max_steps, the iterator yields N batches
  (N = actual length of the stream for one pass).
  Resume:  epochs_trained = gs // U ; steps_trained = (gs % U) * G ;
           skip_first_batches(dl, steps_trained); step = steps_trained-1;
           num_update_steps_trained = steps_trained // G.
-/

def remOrG (S G : Nat) : Nat := if S % G = 0 then G else S % G

/-- One epoch: returns the list of (step index, do_sync_step, window id) for each micro-batch.
    `avail` = number of batches the (possibly skipped) iterator still yields. -/
def runEpoch (G S U avail startU startStep : Nat) : List (Nat × Bool × Nat) :=
  let rec go (u : Nat) (fuel : Nat) (step : Nat) (avail : Nat) : List (Nat × Bool × Nat) :=
    match fuel with
    | 0 => []
    | fuel + 1 =>
      if u ≥ U then [] else
      let nb := if u ≠ U - 1 then G else remOrG S G
      let k := min nb avail
      let evs := (List.range k).map fun i =>
        let s := step + i
        (s, decide ((s + 1) % G = 0 ∨ s + 1 = S), u)
      evs ++ go (u + 1) fuel (step + k) (avail - k)
  go startU U startStep avail

def ceilDiv (a b : Nat) : Nat := (a + b - 1) / b
def nupe (L G : Nat) : Nat := max (L / G + (if L % G > 0 then 1 else 0)) 1

/-- Spec: every optimizer step happens on the last micro-batch of a fetched window and
    the last micro-batch of the epoch triggers a step (no gradient left pending). -/
def windowsOK (evs : List (Nat × Bool × Nat)) : Bool :=
  let rec chk : List (Nat × Bool × Nat) → Bool
    | [] => true
    | [(_, sync, _)] => sync
    | (_, sync, w) :: rest@((_, _, w') :: _) => (sync == (w != w')) && chk rest
  chk evs

-- Map-style: exhaustive check for L ≤ 40, G ≤ 9
#eval Id.run do
  let mut bad : List (Nat × Nat) := []
  for L in List.range 41 do
    for g in List.range 9 do
      let G := g + 1
      if L > 0 && !(windowsOK (runEpoch G L (nupe L G) L 0 0)) then bad := (L, G) :: bad
  return bad

-- Iterable dataset, finite stream of N batches per pass, max_steps M:
--  S = M*G, U = M.  Find (N, G, M) where the last batch of the pass does not trigger
--  an optimizer step, i.e. its gradient leaks into the next pass.
#eval Id.run do
  let mut bad : List (Nat × Nat × Nat) := []
  for N in List.range 12 do
    for g in List.range 4 do
      let G := g + 1
      let M := 10
      if N > 0 && N < M * G && !(windowsOK (runEpoch G (M * G) M N 0 0)) then bad := (N, G, M) :: bad
  return bad.reverse.take 10

#eval runEpoch 2 20 10 5 0 0   -- N=5 batches, G=2, max_steps=10: last batch (step 4) has sync=false

/-- Uninterrupted run: all events over `E` epochs starting fresh, tagged with epoch. -/
def fullRun (G L E : Nat) : List (Nat × Nat × Bool) :=
  (List.range E).flatMap fun e => (runEpoch G L (nupe L G) L 0 0).map fun (s, b, _) => (e, s, b)

/-- global step reached after the first `n` events of a run -/
def gsAfter (evs : List (Nat × Nat × Bool)) (n : Nat) : Nat :=
  ((evs.take n).filter fun (_, _, b) => b).length

/-- Resumed run from global step `gs` (checkpoint taken right after an optimizer step). -/
def resumedRun (G L E gs : Nat) : List (Nat × Nat × Bool) :=
  let U := nupe L G
  let et := gs / U
  let st := (gs % U) * G
  (List.range E).flatMap fun e =>
    if e < et then [] else
    if e == et then (runEpoch G L U (L - st) (st / G) st).map fun (s, b, _) => (e, s, b)
    else (runEpoch G L U L 0 0).map fun (s, b, _) => (e, s, b)

/-- Spec: for every checkpoint position (event index n right after a sync) the resumed run
    replays exactly the remaining events of the uninterrupted run. -/
def resumeOK (G L E : Nat) : Bool :=
  let full := fullRun G L E
  (List.range (full.length + 1)).all fun n =>
    let atSync := n == 0 || (match full[n - 1]? with | some (_, _, b) => b | none => false)
    !atSync || (resumedRun G L E (gsAfter full n) == full.drop n)

#eval Id.run do
  let mut bad : List (Nat × Nat) := []
  for L in List.range 25 do
    for g in List.range 6 do
      let G := g + 1
      if L > 0 && !(resumeOK G L 3) then bad := (L, G) :: bad
  return bad

/-! ### General theorems (all L, G) -/

/-- the sync predicate fires exactly at the last index of the window containing s -/
theorem sync_iff_window_end (G L u r : Nat) (hG : 0 < G) (hr : r < G) (hs : G * u + r < L) :
    ((G * u + r + 1) % G = 0 ∨ G * u + r + 1 = L) ↔ G * u + r + 1 = min (G * u + G) L := by
  have hmod : (G * u + r + 1) % G = (r + 1) % G := by
    rw [Nat.add_assoc, Nat.mul_add_mod]
  rw [hmod]
  by_cases h : r + 1 = G
  · subst h; simp [Nat.mod_self]; omega
  · have : (r + 1) % G = r + 1 := Nat.mod_eq_of_lt (by omega)
    rw [this]; omega

/-- the outer loop's windows tile the epoch exactly: (U-1) full windows + remainder = L -/
theorem windows_tile (L G : Nat) (hG : 0 < G) (hL : 0 < L) :
    (nupe L G - 1) * G + remOrG L G = L := by
  have h := Nat.div_add_mod L G
  have hlt := Nat.mod_lt L hG
  unfold nupe remOrG
  by_cases hm : L % G = 0
  · have hq : 0 < L / G := by
      rcases Nat.eq_zero_or_pos (L / G) with h0 | h0
      · rw [h0] at h; omega
      · exact h0
    simp [hm]
    have : max (L / G) 1 = L / G := by omega
    rw [this]
    have e : (L / G - 1) * G + G = (L / G) * G := by
      rw [Nat.sub_mul, Nat.one_mul]
      have : G ≤ L / G * G := by
        have := Nat.mul_le_mul_right G hq; simpa using this
      omega
    rw [e, Nat.mul_comm]; omega
  · have hpos : L % G > 0 := by omega
    simp only [hm, hpos, ite_true, ite_false]
    have : max (L / G + 1) 1 = L / G + 1 := Nat.max_eq_left (Nat.succ_le_succ (Nat.zero_le _))
    rw [this, Nat.add_sub_cancel, Nat.mul_comm]; omega

/-- batches consumed by the first k update steps when k < U are k*G (all windows full) -/
def consumed (G S U : Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => consumed G S U k + (if k ≠ U - 1 then G else remOrG S G)

theorem consumed_full (G S U k : Nat) (hk : k < U) : consumed G S U k = k * G := by
  induction k with
  | zero => simp [consumed]
  | succ k ih =>
    simp only [consumed]
    rw [ih (by omega)]
    have : k ≠ U - 1 := by omega
    simp [this, Nat.succ_mul]

/-- hence the resume skip `(gs % U) * G` equals the batches the uninterrupted run consumed
    in the current epoch -/
theorem resume_skip_correct (G S U gs : Nat) (hU : 0 < U) :
    consumed G S U (gs % U) = (gs % U) * G :=
  consumed_full G S U (gs % U) (Nat.mod_lt gs hU)

/-- enough epochs are iterated when max_steps is given:  ceil(M/U) * U ≥ M -/
theorem epochs_cover_max_steps (M U : Nat) (hU : 0 < U) :
    (M / U + (if M % U > 0 then 1 else 0)) * U ≥ M := by
  have h := Nat.div_add_mod M U
  have hlt := Nat.mod_lt M hU
  by_cases hm : M % U > 0
  · simp only [hm, ite_true]
    rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]; omega
  · simp only [hm, ite_false]
    rw [Nat.add_zero, Nat.mul_comm]; omega

/-- epoch-based with fractional epochs e = a/b: ceil(e) epochs cover ceil(e*U) steps -/
theorem frac_epochs_cover (a b U : Nat) (hb : 0 < b) :
    ceilDiv a b * U ≥ ceilDiv (a * U) b := by
  unfold ceilDiv
  -- (a*U + b - 1)/b ≤ ((a + b - 1)/b) * U
  have h1 := Nat.div_add_mod (a + b - 1) b
  have h2 := Nat.mod_lt (a + b - 1) hb
  have key : a * U + b - 1 < b * (((a + b - 1) / b) * U) + b := by
    have : a ≤ b * ((a + b - 1) / b) := by omega
    have := Nat.mul_le_mul_right U this
    rw [Nat.mul_assoc] at this
    omega
  have : (a * U + b - 1) / b < ((a + b - 1) / b) * U + 1 := by
    rw [Nat.div_lt_iff_lt_mul hb]
    rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm (((a + b - 1) / b) * U) b]
    exact key
  omega

/-! ### Proposed fix for the iterable leftover: also sync on the last micro-batch of a window that
    came back short from `get_batch_samples` (iterator exhausted), i.e.
      do_sync_step = (step+1) % G == 0 or (step+1) == S or (i == len(batch_samples)-1 and len(batch_samples) < num_batches) -/
def runEpochFixed (G S U avail startU startStep : Nat) : List (Nat × Bool × Nat) :=
  let rec go (u : Nat) (fuel : Nat) (step : Nat) (avail : Nat) : List (Nat × Bool × Nat) :=
    match fuel with
    | 0 => []
    | fuel + 1 =>
      if u ≥ U then [] else
      let nb := if u ≠ U - 1 then G else remOrG S G
      let k := min nb avail
      let evs := (List.range k).map fun i =>
        let s := step + i
        (s, decide ((s + 1) % G = 0 ∨ s + 1 = S ∨ (i + 1 = k ∧ k < nb)), u)
      evs ++ go (u + 1) fuel (step + k) (avail - k)
  go startU U startStep avail

-- fixed version: iterable (any N < M*G) and map-style both satisfy windowsOK
#eval Id.run do
  let mut bad : List (Nat × Nat × Nat) := []
  for N in List.range 30 do
    for g in List.range 6 do
      let G := g + 1
      for M in [1, 3, 10] do
        if N > 0 && N ≤ M * G && !(windowsOK (runEpochFixed G (M * G) M N 0 0)) then bad := (N, G, M) :: bad
  for L in List.range 41 do
    for g in List.range 9 do
      let G := g + 1
      if L > 0 && !(windowsOK (runEpochFixed G L (nupe L G) L 0 0)) then bad := (L, G, 0) :: bad
  return bad
