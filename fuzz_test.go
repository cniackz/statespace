package statespace

import (
	"errors"
	"math"
	"testing"
)

// buildFuzzModel makes a 2-state, 1-observation model from fuzzed entries. Q, R
// and P0 are formed as outer products plus a floor so they are symmetric and
// (for a finite input) positive (semi)definite; F and the observation come
// straight from the fuzz, so extreme and non-finite values reach Validate and
// the filter.
func buildFuzzModel(f0, f1, f2, f3, q0, q1, r0, h0, h1 float64) Model {
	return Model{
		F: FromRows([]float64{f0, f1}, []float64{f2, f3}),
		Q: FromRows([]float64{q0*q0 + 1e-9, q0 * q1}, []float64{q0 * q1, q1*q1 + 1e-9}),
		H: FromRows([]float64{h0, h1}),
		R: FromRows([]float64{r0*r0 + 1e-9}),
		Mu0: Vector{0, 0},
		P0:  Diag(1, 1),
	}
}

// TestExtremeValues is a deterministic sweep of the hard float64 classes through
// Validate and the filter: no panic, a finite result whenever the error is nil,
// and no state left poisoned by a rejected step.
func TestExtremeValues(t *testing.T) {
	hard := []float64{
		0, 1, -1, 1e-320, 5e-324, 1e300, -1e300, 1e308, math.MaxFloat64,
		math.SmallestNonzeroFloat64, math.NaN(), math.Inf(1), math.Inf(-1),
	}
	good := goodModel()
	k, err := NewKF(good)
	if err != nil {
		t.Fatal(err)
	}
	for _, z := range hard {
		stateBefore := k.State()
		covBefore := k.Cov()
		ll, err := k.Update(Vector{z})
		if err != nil {
			// Rejected: nothing may have moved.
			if d := maxVecDiff(stateBefore, k.State()); d != 0 {
				t.Fatalf("z=%g rejected but state moved by %g", z, d)
			}
			if d := maxMatDiff(covBefore, k.Cov()); d != 0 {
				t.Fatalf("z=%g rejected but cov moved by %g", z, d)
			}
			continue
		}
		if math.IsNaN(ll) || math.IsInf(ll, 0) {
			t.Fatalf("z=%g accepted with non-finite loglik %g", z, ll)
		}
		if !vfinite(k.State()) || !k.Cov().isFinite() {
			t.Fatalf("z=%g accepted but left a non-finite estimate", z)
		}
		_ = k.Predict(nil)
	}
}

// FuzzFilter drives Validate, Filter and the online KF with fuzzed model entries
// and observations. The invariants: no panic, a finite result whenever the error
// is nil, and a rejected update leaves the estimate unchanged.
func FuzzFilter(f *testing.F) {
	f.Add(1.0, 0.1, 0.0, 1.0, 0.3, 0.2, 1.0, 1.0, 0.0, 0.5, 0.7)
	f.Add(math.NaN(), 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0)
	f.Add(1e308, 0.0, 0.0, 1e308, 1e300, 0.0, 1.0, 1.0, 0.0, 1e300, -1e300)
	f.Fuzz(func(t *testing.T,
		f0, f1, f2, f3, q0, q1, r0, h0, h1, z0, z1 float64,
	) {
		m := buildFuzzModel(f0, f1, f2, f3, q0, q1, r0, h0, h1)
		if err := m.Validate(); err != nil {
			return // malformed input is rejected, as designed
		}
		k, err := NewKF(m)
		if err != nil {
			return
		}
		for _, z := range []Vector{{z0}, {z1}, nil, {z0 + z1}} {
			stateBefore := k.State()
			covBefore := k.Cov()
			ll, err := k.Update(z)
			if err != nil {
				if !errors.Is(err, ErrInvalidInput) && !errors.Is(err, ErrNotFinite) && !errors.Is(err, ErrInvalidModel) {
					t.Fatalf("unexpected error kind: %v", err)
				}
				if d := maxVecDiff(stateBefore, k.State()); d != 0 {
					t.Fatalf("rejected update moved the state by %g", d)
				}
				if d := maxMatDiff(covBefore, k.Cov()); d != 0 {
					t.Fatalf("rejected update moved the cov by %g", d)
				}
				continue
			}
			if math.IsNaN(ll) || math.IsInf(ll, 0) {
				t.Fatalf("nil error but non-finite loglik %g", ll)
			}
			if !vfinite(k.State()) || !k.Cov().isFinite() {
				t.Fatalf("nil error but non-finite estimate")
			}
			if err := k.Predict(nil); err != nil {
				continue
			}
			if !vfinite(k.State()) || !k.Cov().isFinite() {
				t.Fatalf("predict left a non-finite estimate with nil error")
			}
		}
	})
}
