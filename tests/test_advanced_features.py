# test_advanced_features.py
from pycontinuum import polyvar, PolynomialSystem, ParameterHomotopy, track_parameter_path, solve
import numpy as np


def test_witness_sets():
    """Test computation of witness sets for positive-dimensional components."""
    x, y, z = polyvar('x', 'y', 'z')
    # A line in 3D: x = y, z = 0
    system = PolynomialSystem([x - y, z])
    # Placeholder for future positive-dimensional tests

def test_monodromy():
    """Test monodromy loops for numerical irreducible decomposition."""
    # Placeholder: heavy functionality, skip for unit run

def test_parameter_homotopy_simple_linear():
    """Parameter homotopy should track a known linear solution path."""
    x = polyvar('x')
    # Fixed F(x) = 0 (none for this simple test)
    F = PolynomialSystem([])
    # L1(x) = x - 1, L2(x) = x + 1. Solution moves from x=1 to x=-1 as t goes 0->1
    L1 = PolynomialSystem([x - 1])
    L2 = PolynomialSystem([x + 1])
    ph = ParameterHomotopy(F, L1, L2, [x], square_fix=True)

    start = np.array([1.0 + 0j])
    end, info = track_parameter_path(ph, start_point=start, start_t=0.0, end_t=1.0, options={"tol": 1e-10})

    assert info.get('steps', 0) > 0
    assert np.allclose(end, np.array([-1.0 + 0j]), atol=1e-6)

def test_endgame_singular_corner_case():
    """Ensure solve works on a tiny singular example and returns finite residual solutions."""
    x, y = polyvar('x', 'y')
    system = PolynomialSystem([x**2, x*y])
    # Solve with endgame enabled
    sols = solve(system, variables=[x, y], tol=1e-10, use_endgame=True, verbose=False)
    # We shouldn't crash; residuals should be bounded
    for s in sols:
        assert np.isfinite(s.residual)