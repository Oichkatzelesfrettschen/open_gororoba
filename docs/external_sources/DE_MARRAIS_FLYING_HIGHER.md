# Flying Higher Than A Box-Kite

**Full Title:** Flying Higher Than A Box-Kite: Kite-Chain Middens,
Sand Mandalas, and Zero-Divisor Patterns in the 2^n-ions Beyond the Sedenions

**Author:** Robert P. C. de Marrais (rdemarrais@alum.mit.edu)

**Status:** Unpublished manuscript (no arXiv identifier)

**Source:** convos/CayleyDickson/Pathions2.pdf (305 KB, 15+ pages)

---

## Key Contributions

### 1. The Canonical Strut Table (Page 3)

The strut constant for each box-kite is listed in the first cell of each row.
The Octonion indices of vertices A, B, C are in bold-face (forming the zigzag).
Their Assessor index-pairs are listed in columns A-F.

The "inner XOR" of the Octonion and Sedenion pairs defining the Assessors at
each box-kite vertex will always equal 8 + strut_constant.

| Strut | A     | B     | C     | D     | E     | F     |
|-------|-------|-------|-------|-------|-------|-------|
| 1     | 3, 10 | 6, 15 | 5, 12 | 4, 13 | 7, 14 | 2, 11 |
| 2     | 1, 11 | 7, 13 | 6, 12 | 4, 14 | 5, 15 | 3,  9 |
| 3     | 2,  9 | 5, 14 | 7, 12 | 4, 15 | 6, 13 | 1, 10 |
| 4     | 1, 13 | 2, 14 | 3, 15 | 7, 11 | 6, 10 | 5,  9 |
| 5     | 2, 15 | 4,  9 | 6, 11 | 3, 14 | 1, 12 | 7, 10 |
| 6     | 3, 13 | 4, 10 | 7,  9 | 1, 15 | 2, 12 | 5, 11 |
| 7     | 1, 14 | 4, 11 | 5, 10 | 2, 13 | 3, 12 | 6,  9 |

**Properties:**
- Vertices A, B, C form the zigzag (all-minus-sign sail).
- Their L-indices (the octonion part, listed first) form a NATO triplet.
- The strut pairs are (A, F), (B, E), (C, D) -- strut-opposites never form DMZs.
- Each box-kite excludes exactly one octonion index (1-7), the strut constant.

### 2. Naming Conventions

| Symbol | Dimension | Name      | Origin                     |
|--------|-----------|-----------|----------------------------|
| R      | 1         | Reals     | standard                   |
| C      | 2         | Complex   | standard                   |
| H      | 4         | Hamilton  | standard                   |
| O      | 8         | Octonions | standard                   |
| S      | 16        | Sedenions | standard                   |
| P      | 32        | Pathions  | "32 Paths" of Kabbalah     |
| X      | 64        | Chingons  | 64 hexagrams of I Ching    |
| U      | 128       | Routons   | "Route 128"                |
| V      | 256       | Voudons   | 256 deities of Ifa/Vodou   |

### 3. Pathion (32D) Structure

- Pathions have 15 x 7 = 105 "16-less" triplets of form (O, S, P')
  yielding 210 Assessors in 15 ensembles (sand mandalas).
- In addition, 42 of the (P, P') pairings are ZDs, giving 504 "irreducible" ZDs
  specific to Pathions. With 84 from Sedenions, total is **588** = 3.5x Moreno's count.
- Each sand mandala has 14 Assessors, all sharing the same inner XOR.
- They house 168 fillable cells each (= |PSL(2,7)|).

### 4. Kite-Chain Middens

- Box-kite "harmonics" extend indefinitely into higher 2^n-ions.
- For Sedenion box-kites, harmonics are obtained by adding 16n to assessor indices.
- Two "Assessor harmonics" with indices (o + 16n, S + 16n) and
  (o' + 16n, S' + 16n) will emanate some (o xor o' = S xor S') in the base box-kite.
- The first harmonic (Pathions only) gives 2x the ZD-diagonals of a base box-kite.

### 5. Emanation Tables

- 15 spreadsheet-like layouts (14 x 14) for the Pathion sand mandalas.
- Cell (row, col) = O or S index of the ZD "emanated" by multiplying row-head's
  Assessor times column-head's Assessor, with appropriate edge-sign prefix.
- Both long diagonals are always empty (self-products and strut-opposites).
- Remaining 14^2 - 2*14 = 168 cells fillable.
- For strut constants > 8: only 72 of 168 cells contain emanations.
- For strut constant = 8: "edge polarization" -- peculiar alignment of edge-signs
  by quadrant (all same in upper-left and lower-right, all opposite elsewhere).

### 6. Sand Mandala Patterns (Page 15)

The seven 72-entry tables (with strut constants > 8) display patterns
reminiscent of cellular automata when viewed in sequence. When folded
along row/column midlines, they overlay precisely the vertices of the
Box-Kite whose strut constant equals the "excess" (strut_constant - 8).

### 7. Sedenion Multiplication Table (Page 12)

The 16x16 multiplication table for sedenion basis elements (indices 0-15)
is provided, using a checkerboard shading scheme. Below it, the complete
listing of 155 NATO triplets for the Pathions is given.

### 8. Curvature Tensor Connection (Page 6)

De Marrais notes the box-kite diagram is structurally identical to Milnor's
visualization (from Morse Theory) of the four symmetries of the Riemann
curvature tensor:
1. Skew-symmetry: R(X,Y)Z + R(Y,X)Z = 0
2. Cyclic sum: R(X,Y)Z + R(Y,Z)X + R(Z,X)Y = 0
3. Bracket skew-symmetry: <R(X,Y)Z,W> + <R(X,Y)W,Z> = 0
4. Inner/outer exchange: <R(X,Y)Z,W> = <R(Z,W)X,Y>
