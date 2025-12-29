# nostos-nalgebra

A Nostos extension providing dynamic-sized linear algebra operations using the [nalgebra](https://nalgebra.org/) library.

Unlike `glam` (which provides fixed-size Vec2-4), `nalgebra` supports arbitrary-length vectors and matrices, making it suitable for machine learning, scientific computing, and general linear algebra.

## Features

- **Native handles**: Data is stored directly in nalgebra structures, not copied to/from Nostos lists
- **GC integration**: Memory is automatically freed when handles go out of scope
- **~9x faster** than pure Nostos list operations for vector/matrix math
- **SIMD acceleration**: Leverages CPU vector instructions via [simba](https://docs.rs/simba/)

## Installation

Add to your `nostos.toml`:

```toml
[extensions]
nalgebra = { git = "https://github.com/pegesund/nostos-nalgebra" }
```

## Usage

```nostos
main() = {
    # Create vectors (returns native handle, not a list)
    v1 = __native__("Nalgebra.dvec", [1.0, 2.0, 3.0, 4.0, 5.0])
    v2 = __native__("Nalgebra.dvec", [5.0, 4.0, 3.0, 2.0, 1.0])

    # Vector operations work directly on native handles
    sum = __native__("Nalgebra.dvecAdd", v1, v2)
    dot = __native__("Nalgebra.dvecDot", v1, v2)
    norm = __native__("Nalgebra.dvecNorm", v1)

    println("Dot product: ${dot}")
    println("Norm: ${norm}")

    # Convert back to list when needed
    sumList = __native__("Nalgebra.dvecToList", sum)
    println("Sum: ${sumList}")

    # Dynamic matrices
    m = __native__("Nalgebra.dmatIdentity", 3)
    det = __native__("Nalgebra.dmatDeterminant", m)
    println("Determinant: ${det}")

    0
}
```

## API Reference

### DVector (Dynamic Vectors)

| Function | Description |
|----------|-------------|
| `Nalgebra.dvec(list)` | Create vector from list of floats |
| `Nalgebra.dvecZeros(n)` | Create zero vector of length n |
| `Nalgebra.dvecOnes(n)` | Create vector of ones of length n |
| `Nalgebra.dvecAdd(a, b)` | Add two vectors |
| `Nalgebra.dvecSub(a, b)` | Subtract two vectors |
| `Nalgebra.dvecScale(v, s)` | Multiply vector by scalar |
| `Nalgebra.dvecDiv(a, b)` | Element-wise division |
| `Nalgebra.dvecDot(a, b)` | Dot product |
| `Nalgebra.dvecNorm(v)` | Euclidean norm (length) |
| `Nalgebra.dvecNormalize(v)` | Normalize to unit length |
| `Nalgebra.dvecLen(v)` | Get number of elements |
| `Nalgebra.dvecGet(v, i)` | Get element at index i |
| `Nalgebra.dvecSet(v, i, val)` | Set element (returns new vector) |
| `Nalgebra.dvecMap(a, b)` | Element-wise multiplication |
| `Nalgebra.dvecSum(v)` | Sum of all elements |
| `Nalgebra.dvecMin(v)` | Minimum element |
| `Nalgebra.dvecMax(v)` | Maximum element |
| `Nalgebra.dvecToList(v)` | Convert to Nostos list |

### DMatrix (Dynamic Matrices)

| Function | Description |
|----------|-------------|
| `Nalgebra.dmat(rows)` | Create matrix from nested list |
| `Nalgebra.dmatIdentity(n)` | Create n x n identity matrix |
| `Nalgebra.dmatZeros(rows, cols)` | Create zero matrix |
| `Nalgebra.dmatOnes(rows, cols)` | Create matrix of ones |
| `Nalgebra.dmatFromRows(rows)` | Create from list of row vectors |
| `Nalgebra.dmatFromCols(cols)` | Create from list of column vectors |
| `Nalgebra.dmatAdd(a, b)` | Add two matrices |
| `Nalgebra.dmatSub(a, b)` | Subtract two matrices |
| `Nalgebra.dmatMul(a, b)` | Matrix multiplication |
| `Nalgebra.dmatMulVec(m, v)` | Matrix-vector multiplication |
| `Nalgebra.dmatScale(m, s)` | Multiply matrix by scalar |
| `Nalgebra.dmatTranspose(m)` | Transpose matrix |
| `Nalgebra.dmatRows(m)` | Get number of rows |
| `Nalgebra.dmatCols(m)` | Get number of columns |
| `Nalgebra.dmatGet(m, row, col)` | Get element |
| `Nalgebra.dmatSet(m, row, col, val)` | Set element (returns new matrix) |
| `Nalgebra.dmatGetRow(m, row)` | Get row as vector |
| `Nalgebra.dmatGetCol(m, col)` | Get column as vector |
| `Nalgebra.dmatTrace(m)` | Trace (sum of diagonal) |
| `Nalgebra.dmatDeterminant(m)` | Determinant |
| `Nalgebra.dmatInverse(m)` | Matrix inverse |
| `Nalgebra.dmatDiag(v)` | Create diagonal matrix from vector |
| `Nalgebra.dmatPow(m, n)` | Matrix power |
| `Nalgebra.dmatToList(m)` | Convert to nested Nostos list |

### Utility Functions

| Function | Description |
|----------|-------------|
| `Nalgebra.allocCount()` | Get total allocations (for debugging) |
| `Nalgebra.cleanupCount()` | Get total cleanups (for debugging) |
| `Nalgebra.resetStats()` | Reset allocation counters |

## Performance

Native handles provide significant performance improvements over pure Nostos list operations:

| Operation | Native Handles | Pure Nostos | Speedup |
|-----------|---------------|-------------|---------|
| Vector add + sum (5 elem) | ~30μs | ~280μs | 9x |
| Dot product (10 elem) | ~2μs | ~22μs | 10x |
| Large vector ops (100 elem) | ~60μs | ~558μs | 9x |

The speedup comes from:
- No data copying between Nostos and Rust
- SIMD-accelerated operations in nalgebra
- Efficient memory layout for numeric data

## License

MIT
