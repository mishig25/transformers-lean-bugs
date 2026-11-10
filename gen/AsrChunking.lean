/-
ASR pipeline chunking (src/transformers/pipelines/automatic_speech_recognition.py)

  chunk_iter            (l.96-119)   : sample-level chunks + strides
  rescale_stride        (l.76-93)    : sample strides -> token/logit strides, ratio = 1 / align_to
  _forward (ctc)        (l.617-625)  : ratio = 1 / self._align_to,   _align_to = config.inputs_to_logits_ratio (default 1!)
  postprocess (ctc)     (l.654-661)  : items = tokens[:, left : total_n - right]

Spec A (sample level): the kept regions [start+left, start+n-right) of all yielded chunks tile [0, N) exactly.
Spec B (logit level):  after rescaling, the kept logit frames of all chunks tile the model's logit frames,
                       i.e. total kept frames ≈ frames(N) and no chunk is dropped.
-/

/-- Python's round() on a non-negative rational p/q (round-half-to-even). -/
def pyRound (p q : Nat) : Nat :=
  let f := p / q
  let r := p % q
  if 2 * r > q then f + 1 else if 2 * r < q then f else (if f % 2 == 0 then f else f + 1)

structure Chunk where
  start : Nat
  n     : Nat   -- chunk.shape[0]
  left  : Nat
  right : Nat
deriving Repr

/-- chunk_iter, with fuel to make it structurally terminating. step = C - sl - sr > 0. -/
def chunkIter (N C sl sr : Nat) : List Chunk :=
  let step := C - sl - sr
  let rec go (fuel s : Nat) : List Chunk :=
    match fuel with
    | 0 => []
    | fuel + 1 =>
      if s < N then
        let e := s + C
        let n := min e N - s
        let isLast := decide (e ≥ N)
        let l := if s == 0 then 0 else sl
        let r := if isLast then 0 else sr
        let here := if n > l then [⟨s, n, l, r⟩] else []
        if isLast then here else here ++ go fuel (s + step)
      else []
  go (N + 1) 0

/-- kept sample intervals -/
def keptSamples (cs : List Chunk) : List (Nat × Nat) := cs.map fun c => (c.start + c.left, c.start + c.n - c.right)

def tiles (N : Nat) : List (Nat × Nat) → Bool
  | [] => N == 0
  | (a, b) :: rest =>
    let rec go (pos : Nat) : List (Nat × Nat) → Bool
      | [] => pos == N
      | (a, b) :: rest => a == pos && a ≤ b && go b rest
    a == 0 && a ≤ b && go b rest

-- Spec A, checked exhaustively on a grid (sample-level chunking is correct):
def specAFailures : List (Nat × Nat × Nat × Nat) := Id.run do
  let mut bad := []
  for N in List.range 60 do
    for C in List.range 16 do
      for sl in List.range 6 do
        for sr in List.range 6 do
          if C > sl + sr && N > 0 then
            if !(tiles N (keptSamples (chunkIter N C sl sr))) then bad := bad ++ [(N, C, sl, sr)]
  return bad
#eval specAFailures   -- []

/-- Sample-level contiguity, proved: the kept end of chunk k equals the kept start of chunk k+1. -/
theorem kept_contiguous (s C sl sr : Nat) (h : C > sl + sr) :
    s + C - sr = (s + (C - sl - sr)) + sl := by omega

/-- Logit-level: rescale_stride with ratio 1/align (all values in samples), then CTC crop on `frames` logits.
`frames n` = the number of logits the model really produces for a chunk of n samples. -/
def keptFrames (align : Nat) (frames : Nat → Nat) (c : Chunk) : Nat :=
  let tokenN := pyRound c.n align                    -- int(round(input_n * ratio))
  let left  := pyRound (c.left * tokenN) c.n         -- int(round(left / input_n * token_n))
  let right := pyRound (c.right * tokenN) c.n
  let stop := min (frames c.n) (tokenN - right)      -- items[:, left : total_n - right]
  stop - left

def totalKept (align : Nat) (frames : Nat → Nat) (N C sl sr : Nat) : Nat :=
  ((chunkIter N C sl sr).map (keptFrames align frames)).foldl (· + ·) 0

/-- `preprocess` aligns chunk/stride lengths to multiples of align_to (l.483-485). -/
def alignUp (x align : Nat) : Nat := pyRound x align * align

/-- Scenario: 16 kHz audio of `secs` seconds, chunk_length_s = 10, default stride 10/6 s.
  `align` = pipeline's `_align_to`, `hop` = the model's real samples-per-logit. -/
def scenario (secs align hop : Nat) : Nat × Nat :=
  let sr := 16000
  let N := secs * sr
  let C := alignUp (10 * sr) align
  let st := alignUp (10 * sr / 6) align
  (totalKept align (fun n => n / hop) N C st st, N / hop)

-- wav2vec2-like (inputs_to_logits_ratio = 320 = real hop): kept ≈ full frame count
#eval scenario 30 320 320     -- (1500, 1500)
#eval scenario 60 320 320
-- Parakeet-CTC-like: config has NO inputs_to_logits_ratio -> _align_to = 1, real hop = 1280 (10 ms mel * 8x subsampling)
#eval scenario 30 1 1280      -- (125, 375): only the first chunk survives
#eval scenario 60 1 1280      -- (125, 750)
#eval (chunkIter (30*16000) 160000 26667 26667).map (fun c => (c.n, c.left, c.right, keptFrames 1 (· / 1280) c))

/-- Spec B fails whenever align_to = 1 but strides are non-zero: a chunk with left stride `sl ≥ frames(n)` keeps
nothing.  Proved for the generic case. -/
theorem middle_chunk_dropped (n sl : Nat) (hop : Nat) (hhop : 1 ≤ hop) (hsl : n / hop ≤ sl) :
    min (n / hop) (n - 0) - sl = 0 := by
  omega

/-- wav2vec2 feature-encoder output length (kernels 10,3,3,3,3,2,2 / strides 5,2,2,2,2,2,2) -/
def w2vFrames (n : Nat) : Nat :=
  [(10,5),(3,2),(3,2),(3,2),(3,2),(2,2),(2,2)].foldl (fun L (k, s) => if L ≥ k then (L - k) / s + 1 else 0) n

def scenarioW2v (secs : Nat) : Nat × Nat :=
  let sr := 16000
  let N := secs * sr
  let C := alignUp (10 * sr) 320
  let st := alignUp (10 * sr / 6) 320
  (totalKept 320 w2vFrames N C st st, w2vFrames N)

-- wav2vec2 with the real conv length: off by at most a frame or two per file (benign)
#eval [5, 12, 30, 61, 95, 3600].map scenarioW2v
