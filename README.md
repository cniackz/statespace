# statespace

`statespace` is a small Go library, with no dependencies beyond the standard
library, for the linear Gaussian state-space model of arbitrary dimension:

    x_k = F x_{k-1} + B u_k + w_k,   w_k ~ N(0, Q)
    z_k = H x_k + v_k,               v_k ~ N(0, R)

with the prior x_0 ~ N(Mu0, P0). It gives you a numerically stable filter, a
smoother, and maximum-likelihood parameter learning.

```go
m := statespace.Model{
    F:   statespace.FromRows([]float64{1, 1}, []float64{0, 1}),
    Q:   statespace.Diag(1e-4, 1e-4),
    H:   statespace.FromRows([]float64{1, 0}),
    R:   statespace.FromRows([]float64{1}),
    Mu0: statespace.Vector{0, 0},
    P0:  statespace.Diag(1, 1),
}
res, err := m.Filter(zs, nil)      // zs []Vector; a nil entry is a gap
sm, err := m.Smooth(zs, nil)       // fixed-interval smoothing
fit, err := statespace.Learn(zs, m) // fit F, Q, H, R, Mu0, P0 by EM
```

## What it does

- A **Kalman filter** carried in UD (modified-Cholesky) factored form, so the
  covariance stays symmetric and positive semidefinite in finite precision and
  the filter does not diverge from a loss of definiteness. It is the measurement
  update of Bierman (1977) and the time update of Thornton (1976), together the
  UDU^T filter.
- The **Rauch-Tung-Striebel** fixed-interval smoother, which refines every
  estimate using the whole series, and returns the lag-one smoothed covariances.
- **Parameter learning by EM** for the linear dynamical system: one forward
  filter pass and one backward smoother pass per iteration, with closed-form
  updates for F, Q, H, R, Mu0 and P0. The log-likelihood is non-decreasing
  across iterations.
- An **online filter** (`KF`) with `Predict` and `Update` for streaming, control
  inputs through `B`, complete gaps (`Update(nil)`) and partial gaps
  (`UpdateMasked`), and validation that rejects a malformed model rather than
  returning quiet garbage.

The math and its sources are in the package godoc: Kalman (1960); Rauch, Tung
and Striebel (1965); Bierman (1977); Thornton (1976); Dempster, Laird and Rubin
(1977); Shumway and Stoffer (1982); Ghahramani and Hinton (1996).

## Why UD and not the textbook update

The naive covariance update P = (I - K H) P loses symmetry and
positive-definiteness after enough steps and can make P indefinite, at which
point the filter diverges. The UD factorization carries P = U D U^T and never
forms P, so D stays non-negative by construction. The filter is validated
against an independent Joseph-form reference: on 200 random systems, including
sequences with gaps, the filtered mean and covariance agree to 1e-8 and the
log-likelihood to 1e-7.

## Correctness

Measured with `go test`:

- The UD filter matches the Joseph-form reference in mean, covariance and
  log-likelihood (`TestUDMatchesJoseph`), and the reconstructed covariance stays
  symmetric positive semidefinite after every step (`TestCovStaysPSD`).
- The smoother never increases the covariance over the filter
  (`TestSmootherBeatsFilter`), and on a seeded constant-velocity system it
  recovers the hidden position with RMSE 0.37 against a measurement standard
  deviation of 1.0 (`TestSmootherOracle`).
- EM's log-likelihood is non-decreasing across iterations (`TestEMMonotone`),
  and with the dynamics known it recovers the process and measurement variances
  from a wrong start to about 1% on a long series (`TestEMRecoversParams`).
- Full-float64-class fuzzing of the model entries and observations finds no
  panic, no non-finite result returned without an error, and no state left
  changed by a rejected step (`FuzzFilter`; over ten million executions with no
  failure).

## Performance

`go test -run xxx -bench . -benchmem` (Apple M4 Pro, state dimension 4):

| op | time | allocations |
|---|---|---|
| `KF.Predict` | about 0.6 us | 6 |
| `KF.Update` | about 0.8 us | 10 |
| `Filter` (100 steps) | about 215 us | 2017 |
| `Smooth` (100 steps) | about 265 us | 3802 |

The library is dense and O(n^3) per step in the state dimension n, O(T n^3) over
a series of length T, and is meant for moderate n. The online step is not
allocation-free: it validates the next state in scratch and commits only if it
is finite, so a rejected step leaves the estimate untouched, and that safety
costs a few allocations per call. If you need a zero-allocation hot loop at a
fixed small dimension, a specialized filter (such as a two-state tracker) will
beat this one; `statespace` trades that for arbitrary dimension and the
validate-before-commit guarantee.

## Known limits

- EM converges to a local optimum and is sensitive to the initial model.
  `Learn` returns the full log-likelihood trace so you can see the climb and
  compare restarts; start from a reasonable guess.
- The scale of the state and the sign of a state component are identifiable only
  up to the usual ambiguities of the linear Gaussian model, so a learned F and Q
  can differ from a data-generating pair that an H absorbs. The likelihood, not
  the individual matrices, is what EM maximizes; fix what you know (for example a
  known H) to pin the rest.
- A partial gap is requested explicitly with a mask (`UpdateMasked`); a NaN in an
  observation that is not a declared gap is invalid input, not a gap.
- `Learn` does not estimate a control model (`B`); `Filter` and `Smooth` do take
  control inputs.
- The types are not safe for concurrent use without external locking.

## License

Apache 2.0. See `LICENSE` and `NOTICE`.
