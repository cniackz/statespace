package statespace

import (
	"math"
	"testing"
)

// TestInvertPredictedIllConditioned covers attack N2: a predicted covariance
// that is merely ill-conditioned (a real but small-variance direction, mixed
// state units) must keep its full inverse, not have the small direction dropped.
func TestInvertPredictedIllConditioned(t *testing.T) {
	for _, small := range []float64{1e-6, 1e-10, 1e-14} {
		pp := Diag(1e2, small)
		inv := invertPredicted(pp)
		want := Diag(1e-2, 1/small)
		for i := 0; i < 2; i++ {
			got := inv.At(i, i)
			if d := math.Abs(got-want.At(i, i)) / want.At(i, i); d > 1e-8 {
				t.Errorf("small=%g: inv[%d][%d]=%g, want %g (rel %g)", small, i, i, got, want.At(i, i), d)
			}
		}
	}
}

// TestInvertPredictedSingular covers the deterministic case (rank-deficient Q):
// a zero-variance direction gets no inverse (zero row and column), and the rest
// is inverted.
func TestInvertPredictedSingular(t *testing.T) {
	pp := Diag(0, 0.1)
	inv := invertPredicted(pp)
	if inv.At(0, 0) != 0 || inv.At(0, 1) != 0 || inv.At(1, 0) != 0 {
		t.Errorf("deterministic direction not dropped: %v", inv.data)
	}
	if d := math.Abs(inv.At(1, 1) - 10); d > 1e-9 {
		t.Errorf("inv[1][1]=%g, want 10", inv.At(1, 1))
	}
	for _, v := range inv.data {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite pseudo-inverse: %v", inv.data)
		}
	}
}

// TestValidateRejectsMixedScaleIndefinite covers attack N1b: a small indefinite
// block next to a large one is rejected, because the correlation exposes it even
// though its entries are tiny next to the large block.
func TestValidateRejectsMixedScaleIndefinite(t *testing.T) {
	q := FromRows(
		[]float64{1e10, 0, 0},
		[]float64{0, 1e-12, 1e-10},
		[]float64{0, 1e-10, 1e-12},
	)
	m := Model{F: Ident(3), Q: q, H: FromRows([]float64{1, 0, 0}), R: Diag(1), Mu0: Vector{0, 0, 0}, P0: Ident(3)}
	if err := m.Validate(); err == nil {
		t.Error("mixed-scale indefinite Q accepted")
	}
}

// TestValidateRejectsSubnormalIndefinite covers attack N3: an indefinite matrix
// at a subnormal scale is rejected, not accepted through a NaN that slips past
// the comparison, and the eigenvalue routine sees the indefiniteness.
func TestValidateRejectsSubnormalIndefinite(t *testing.T) {
	tiny := math.SmallestNonzeroFloat64 * 4
	if e := minEigSym(Diag(tiny, -tiny)); !(e < 0) {
		t.Errorf("minEigSym(diag(%g,%g))=%g, want negative", tiny, -tiny, e)
	}
	m := Model{F: Ident(2), Q: Diag(tiny, -tiny), H: Ident(2), R: Diag(1, 1), Mu0: Vector{0, 0}, P0: Ident(2)}
	if err := m.Validate(); err == nil {
		t.Error("subnormal indefinite Q accepted")
	}
}

// TestMinEigScaleInvariantExtreme pins the scale invariance of the eigenvalue
// routine from subnormal to huge, where dividing by the raw scale would overflow
// or lose the result (attack N3 and the H3 family).
func TestMinEigScaleInvariantExtreme(t *testing.T) {
	a := FromRows([]float64{0.1, 1}, []float64{1, 0.1}) // eigenvalues 1.1 and -0.9
	for _, s := range []float64{1e-300, 1e-150, 1, 1e150, 1e300} {
		e := minEigSym(scaleMat(a, s))
		if d := math.Abs(e-(-0.9*s)) / (0.9 * s); d > 1e-9 {
			t.Errorf("scale %g: minEig %g, want %g (rel %g)", s, e, -0.9*s, d)
		}
	}
}

// TestValidateAcceptsIllConditioned is the counterpart: a genuinely valid
// covariance that is only ill-conditioned (a wide but positive spread of
// variances) must pass, so the stricter checks do not reject real models.
func TestValidateAcceptsIllConditioned(t *testing.T) {
	m := Model{
		F: Ident(2), Q: Diag(1e2, 1e-14), H: Ident(2),
		R: Diag(1, 1e-10), Mu0: Vector{0, 0}, P0: Diag(1e2, 1e-14),
	}
	if err := m.Validate(); err != nil {
		t.Errorf("ill-conditioned but valid model rejected: %v", err)
	}
}

// TestInvertPredictedOffAxisSingular covers the smoother inverse on a singular
// predicted covariance whose null direction is not a coordinate axis (a
// rank-deficient Q in a rotated basis). The ordinary inverse returns a finite
// but wrong matrix there, which only the residual check rejects, so this is what
// makes that check load-bearing. The pseudo-inverse must satisfy the reflexive
// identity P inv P = P, which the wrong ordinary inverse does not.
func TestInvertPredictedOffAxisSingular(t *testing.T) {
	pp := FromRows([]float64{1e8, 1e4}, []float64{1e4, 1}) // singular: 1e8*1 = (1e4)^2
	inv := invertPredicted(pp)
	for _, v := range inv.data {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite inverse: %v", inv.data)
		}
	}
	back := matMul(matMul(pp, inv), pp)
	var scale float64
	for _, v := range pp.data {
		if a := math.Abs(v); a > scale {
			scale = a
		}
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if d := math.Abs(back.At(i, j) - pp.At(i, j)); d > 1e-6*scale {
				t.Errorf("P inv P != P at [%d][%d]: %g vs %g", i, j, back.At(i, j), pp.At(i, j))
			}
		}
	}
}
