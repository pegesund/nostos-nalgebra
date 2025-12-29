# nostos-nalgebra

A Nostos extension providing dynamic-sized linear algebra operations using the [nalgebra](https://nalgebra.org/) library.

Unlike `glam` (which provides fixed-size Vec2-4), `nalgebra` supports arbitrary-length vectors and matrices, making it suitable for machine learning, scientific computing, and general linear algebra.

## Installation

Add to your `nostos.toml`:

```toml
[extensions]
nalgebra = { git = "https://github.com/pegesund/nostos-nalgebra" }
```

## Usage

```nostos
import nalgebra

main() = {
    # Dynamic vectors of any length
    v1 = nalgebra.dvec([1.0, 2.0, 3.0, 4.0, 5.0])
    v2 = nalgebra.dvec([5.0, 4.0, 3.0, 2.0, 1.0])

    # Vector operations
    sum = nalgebra.dvecAdd(v1, v2)
    dot = nalgebra.dvecDot(v1, v2)
    norm = nalgebra.dvecNorm(v1)

    println("Sum: ${sum}")
    println("Dot product: ${dot}")
    println("Norm: ${norm}")

    # Dynamic matrices
    m = nalgebra.dmat([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0]
    ])

    det = nalgebra.dmatDeterminant(m)
    inv = nalgebra.dmatInverse(m)

    println("Determinant: ${det}")
    println("Inverse: ${inv}")

    0
}
```

## API Reference

### DVector (Dynamic Vectors)

| Function | Description |
|----------|-------------|
| `dvec(list)` | Create vector from list |
| `dvecZeros(n)` | Create zero vector of length n |
| `dvecOnes(n)` | Create vector of ones of length n |
| `dvecAdd(a, b)` | Add two vectors |
| `dvecSub(a, b)` | Subtract two vectors |
| `dvecScale(v, s)` | Multiply vector by scalar |
| `dvecDot(a, b)` | Dot product |
| `dvecNorm(v)` | Euclidean norm (length) |
| `dvecNormalize(v)` | Normalize to unit length |
| `dvecLen(v)` | Get number of elements |
| `dvecGet(v, i)` | Get element at index i |
| `dvecSet(v, i, val)` | Set element (returns new vector) |
| `dvecComponentMul(a, b)` | Element-wise multiplication |
| `dvecSum(v)` | Sum of all elements |
| `dvecMin(v)` | Minimum element |
| `dvecMax(v)` | Maximum element |
| `dvecDistance(a, b)` | Distance between points |
| `dvecLerp(a, b, t)` | Linear interpolation |

### DMatrix (Dynamic Matrices)

| Function | Description |
|----------|-------------|
| `dmat(rows)` | Create matrix from nested list |
| `dmatIdentity(n)` | Create n x n identity matrix |
| `dmatZeros(rows, cols)` | Create zero matrix |
| `dmatOnes(rows, cols)` | Create matrix of ones |
| `dmatFromRows(rows)` | Create from list of row vectors |
| `dmatFromCols(cols)` | Create from list of column vectors |
| `dmatAdd(a, b)` | Add two matrices |
| `dmatSub(a, b)` | Subtract two matrices |
| `dmatMul(a, b)` | Matrix multiplication |
| `dmatMulVec(m, v)` | Matrix-vector multiplication |
| `dmatScale(m, s)` | Multiply matrix by scalar |
| `dmatTranspose(m)` | Transpose matrix |
| `dmatRows(m)` | Get number of rows |
| `dmatCols(m)` | Get number of columns |
| `dmatGet(m, row, col)` | Get element |
| `dmatSet(m, row, col, val)` | Set element (returns new matrix) |
| `dmatGetRow(m, row)` | Get row as vector |
| `dmatGetCol(m, col)` | Get column as vector |
| `dmatTrace(m)` | Trace (sum of diagonal) |
| `dmatDeterminant(m)` | Determinant |
| `dmatInverse(m)` | Matrix inverse |
| `dmatDiag(v)` | Create diagonal matrix from vector |
| `dmatPow(m, n)` | Matrix power |
| `dmatIsSquare(m)` | Check if square |
| `dmatShape(m)` | Get (rows, cols) tuple |

## SIMD Support

nalgebra uses the [simba](https://docs.rs/simba/) crate for SIMD acceleration. This means batch operations on vectors and matrices can leverage SIMD instructions for improved performance.

## License

MIT
