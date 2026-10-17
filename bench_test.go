package statespace

import (
	"math/rand"
	"testing"
)

func benchModel(n int) Model {
	rng := rand.New(rand.NewSource(1))
	return randModel(rng, n, n)
}

func BenchmarkPredict(b *testing.B) {
	k, _ := NewKF(benchModel(4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = k.Predict(nil)
	}
}

func BenchmarkUpdate(b *testing.B) {
	k, _ := NewKF(benchModel(4))
	z := Vector{0.1, 0.2, 0.3, 0.4}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = k.Update(z)
	}
}

func BenchmarkFilter100(b *testing.B) {
	m := benchModel(4)
	rng := rand.New(rand.NewSource(2))
	zs := simulate(rng, m, 100, nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = m.Filter(zs, nil)
	}
}

func BenchmarkSmooth100(b *testing.B) {
	m := benchModel(4)
	rng := rand.New(rand.NewSource(3))
	zs := simulate(rng, m, 100, nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = m.Smooth(zs, nil)
	}
}
