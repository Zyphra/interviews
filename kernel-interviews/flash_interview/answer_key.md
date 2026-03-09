# Flash Attention Interview — Answer Key & Grading Rubric

## Overview

- **Position:** Kernel Engineer
- **Duration:** 60 minutes
- **GPU:** T4 (SM75) on Google Colab
- **Based on:** Colfax CUTLASS FlashAttention-2 repository

---

## Correct Solutions

### TASK A: Load Q, K, V tiles into shared memory (25 pts)

```cuda
// Load Q tile
for (int k = 0; k < d; k++) {
    sQ[tid][k] = (qRow < N) ? h2f(Q[bhOffset + qRow * d + k]) : 0.0f;
}

// Load K and V tiles (inside the kvTile loop)
int kvRow = kvStart + tid;
for (int k = 0; k < d; k++) {
    sK[tid][k] = (kvRow < N) ? h2f(K[bhOffset + kvRow * d + k]) : 0.0f;
    sV[tid][k] = (kvRow < N) ? h2f(V[bhOffset + kvRow * d + k]) : 0.0f;
}
```

**Grading:**
- 15 pts: Correct indexing (bhOffset + row * d + k) for Q, K, V
- 5 pts:  Proper boundary checks (qRow < N, kvRow < N) with zero-padding
- 5 pts:  Uses h2f() for half → float conversion

**Common mistakes:**
- Forgetting boundary checks → garbage values, wrong results
- Wrong stride calculation (e.g., using N instead of d)
- Forgetting h2f() conversion (shared mem is float, global is half)

---

### TASK B: Online Softmax (50 pts)

```cuda
// Step 1: Find new max
float m_new = row_max;
for (int j = 0; j < BLK_N; j++) {
    m_new = fmaxf(m_new, scores[j]);
}

// Step 2-4: Rescale old accumulator to new max
float alpha = expf(row_max - m_new);
row_sum *= alpha;
for (int k = 0; k < d; k++) {
    acc[k] *= alpha;
}

// Step 5: Accumulate new tile
for (int j = 0; j < BLK_N; j++) {
    float p = expf(scores[j] - m_new);
    row_sum += p;
    for (int k = 0; k < d; k++) {
        acc[k] += p * sV[j][k];
    }
}

// Step 6: Update running max
row_max = m_new;
```

**Grading:**
- 10 pts: Correctly finds new max across scores AND old row_max
- 15 pts: Correct rescaling factor `alpha = exp(old_max - new_max)`
          and applies it to BOTH row_sum and acc[]
- 15 pts: Correct accumulation: exp(score - m_new), adds to row_sum,
          weighted sum into acc[]
- 10 pts: Updates row_max = m_new at the end

**Common mistakes:**
- Forgetting to include row_max in the max computation (only taking max of scores)
- Forgetting to rescale acc[] (only rescaling row_sum)
- Using exp(scores[j]) instead of exp(scores[j] - m_new) — numerical instability
- Dividing by row_sum inside the loop (should defer to the end)
- Not updating row_max after processing the tile

**Exceptional answers (bonus discussion points):**
- Candidate mentions exp2f() is faster than expf() on NVIDIA GPUs
- Candidate notes that for first tile (row_max == -inf), alpha = 0, which
  correctly zeroes out acc (no need for special-casing)
- Candidate discusses log-sum-exp stability guarantees

---

### TASK C: Final normalization and write-out (25 pts)

```cuda
if (qRow < N) {
    float inv_sum = 1.0f / row_sum;
    for (int k = 0; k < d; k++) {
        O[bhOffset + qRow * d + k] = f2h(acc[k] * inv_sum);
    }
}
```

**Grading:**
- 10 pts: Divides acc by row_sum (not forgetting this step)
- 10 pts: Correct global memory indexing and f2h() conversion
- 5 pts:  Boundary check (qRow < N)

**Common mistakes:**
- Writing without boundary check → out-of-bounds writes
- Forgetting f2h() conversion (output is half)
- Wrong output indexing

---

## Discussion Questions — Expected Answers

### 1. Bank Conflicts
- sK has 64 floats per row = 256 bytes. 32 banks × 4 bytes = 128 bytes per bank cycle.
- When all threads access sK[j][k] for same j but sequential k, consecutive threads
  hit consecutive banks → **no bank conflicts** for this access pattern.
- However, the dot product loop has all threads reading the SAME sK[j][k] for each k
  (broadcast), which is fine (broadcast is handled without conflicts on SM75+).
- A candidate who mentions **padding** (e.g., sK[BLK_N][BLK_D+1]) to avoid conflicts
  in other access patterns shows strong understanding.

### 2. Numerical Stability
- exp(x) overflows for x > ~88 (float). Subtracting max ensures exponents are ≤ 0.
- FP16 has only ~3.3 decimal digits of precision; accumulating many small exp values
  in FP16 would lose precision catastrophically. The repo always uses float for softmax.
- exp2f(x * log2e) maps to a single PTX instruction (`ex2.approx.f32`) which is faster
  than expf() which requires a range reduction + polynomial approximation.

### 3. Production Optimizations
- **Software pipelining:** Overlaps memory loads of tile N+1 with computation on tile N.
  TMA loads are asynchronous, so the warp can issue a load and immediately start computing.
  With multiple stages/buffers, you can have several loads in flight.
- **Warp specialization:** Producer warps handle only TMA loads (simpler register
  pressure), consumer warps handle only MMA (can allocate more registers for accumulators).
  The CUTLASS repo's fmha_pipe_ws.h uses `warpgroup_reg_dealloc<80>` for producers and
  `warpgroup_reg_alloc<192>` for consumers — this asymmetric register allocation is only
  possible with warp specialization.
- **Multi-buffering (STAGES=3):** With 2 buffers, you can overlap 1 load with 1 compute.
  With 3 buffers, you can tolerate longer memory latencies — while computing on buffer 0,
  buffer 1's load completes, and buffer 2's load is in progress. This keeps the compute
  pipeline fully utilized even with high-latency global memory accesses.

### 4. Scaling
- BLK_M=128 → 128 threads per block, 128×64×4 = 32KB for sQ alone. With sK and sV,
  total smem ≈ 32KB × 3 = 96KB, which exceeds T4's 64KB smem per block (or requires
  dynamic smem and reduces occupancy).
- Larger tiles reduce kernel launch overhead and improve data reuse, but hurt occupancy.
- Production kernels tune tile sizes per GPU: Hopper has 228KB smem, so 128×128 tiles
  are feasible. The repo offers QBLKSIZE and KBLKSIZE as compile-time macros.

---

## Bonus: Causal Masking

Add inside the score computation loop, after computing `scores[j]`:
```cuda
// Causal mask: query at position qRow can only attend to keys at positions ≤ qRow
if (kvStart + j > qRow) scores[j] = -FLT_MAX;
```

And optionally, skip entire tiles where `kvStart >= qRow + 1` (early exit):
```cuda
if (kvStart > qRow) break;  // All keys in this and subsequent tiles are masked
```

---

## Scoring Summary

| Component | Points | Notes |
|-----------|--------|-------|
| Task A    | 25     | Shared memory loads with correct indexing |
| Task B    | 50     | Online softmax — the core algorithmic test |
| Task C    | 25     | Final normalization and write-out |
| **Total** | **100** | |

**Pass threshold:** 70+ points (Tasks A + C correct, Task B mostly correct)

**Strong hire signal:** 85+ points, good discussion answers, mentions optimization ideas

**Additional positive signals:**
- Asks clarifying questions about memory layout before coding
- Tests edge cases mentally (N not divisible by BLK_N, first tile special case)
- Mentions that this thread-per-row approach doesn't use tensor cores and
  describes how MMA instructions would change the design
- Discusses register pressure tradeoffs
