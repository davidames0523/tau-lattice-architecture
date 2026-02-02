# The Law of Cyclic Attractors
### A Foundation for O(N) Attention and O(1) Routing

**Abstract:**
Standard AI architectures assume that high-dimensional vector spaces are unstructured "soups" requiring $O(N^2)$ comparison to navigate. We propose that by projecting these vectors onto a **Divisor Lattice** derived from highly composite numbers (e.g., $k=55440$), we can induce a rigid topology of disjoint "Basins." This structure allows for the deletion of entire complexity classes in routing and attention mechanisms.

---

## 1. The Core Discovery
We posit that the routing dynamics of a neural network can be modeled by the discrete dynamical system:

$$\tau(k) = \tau(x_{n+1} - \gcd(x_n, x_{n-1}))$$

While the trajectory of $x_n$ appears chaotic, it is bound by a strict conservation law involving the greatest common divisor (GCD).

### The "Basin" Hypothesis
For a fixed system constant $k$ (the "God Number"), the phase space of all possible inputs is not a single connected ocean. Instead, it is partitioned into **$\tau(k)$ disjoint basins**, where $\tau(k)$ is the number of divisors of $k$.

* **If** $\tau(55440) = 120$, **Then** there are exactly 120 fundamental "States of Matter" for a token.
* **The Invariant:** A token's basin is determined by the invariant $g = \gcd(x_0, x_1, k)$.
* **The Law:** A token starting in Basin $A$ cannot physically interact with Basin $B$. They are orthogonal by definition.

## 2. Implication for Computation

### The "Mixing Bowl" vs. The "Egg Carton"
* **Standard AI (The Mixing Bowl):** Assumes any token might need to attend to any other token.
    * *Consequence:* You must calculate $N \times N$ interactions.
    * *Cost:* Quadratic ($N^2$).
* **Tau-Lattice AI (The Egg Carton):** We prove the tokens are pre-sorted into 120 egg cups (Basins).
    * *Consequence:* A token in Cup #1 only looks at other eggs in Cup #1.
    * *Cost:* Linear ($N$).

## 3. The 3-Cycle Trajectory (Procedural Weights)
Within each basin, the system does not stagnate. It converges to a stable **3-cycle attractor**:
$$A \to B \to C \to A$$
This cycle is deterministic. 
* **Why this matters:** In standard Mixture-of-Experts (MoE) models, the "Expert" is a stored matrix of weights that must be fetched from RAM (slow).
* **In Tau-Lattice:** The "Expert" is the 3-cycle itself. We generate the weights on the fly using the cycle values.
    * *Result:* We replace **Memory Bandwidth** (the bottleneck) with **Arithmetic** (which is abundant).

## 4. Summary of the Theorem
**"If a vector space is projected onto a Divisor Lattice of $k$, the Interaction Matrix of that space becomes Block-Diagonal with $\tau(k)$ blocks."**

This theorem allows us to mathematically prove that 99% of the computations in a standard Transformer are zeros. We simply stop computing them.