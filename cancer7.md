#### a)
Strang Splitting 的對稱結構：
* `u = self.reaction_exact(u, dt/2)` （半步反應項 $\exp(R \frac{\Delta t}{2})$）
* `u_hat_new = u_hat * decay_factor + f_hat * factor` （一步擴散項 $\exp(D \Delta t)$）
* `u = self.reaction_exact(u, dt/2)` （半步反應項 $\exp(R \frac{\Delta t}{2})$）

<br></br>


$$\text{Local Truncation Error} = \mathcal{O}(\Delta t^3)$$
$$\text{Global Truncation Error} = \mathcal{O}(\Delta t^2)$$

Proof:

$$ \frac{du}{dt} = (A + B)u \quad\text{,where A: Diffusion and B:  Reaction} $$  

,where A:Diffusion & B: Reaction

The exact solution after one time step $\Delta t$ is:

$$u_{exact}(\Delta t) = e^{(A+B)\Delta t} u_0$$

Expanding the exponential operator using the Taylor series:

$$e^{(A+B)\Delta t} = I + (A+B)\Delta t + \frac{1}{2}(A+B)^2 \Delta t^2 + \frac{1}{6}(A+B)^3 \Delta t^3 + \mathcal{O}(\Delta t^4)$$

Expanding the quadratic term $(A+B)^2$:

$$(A+B)^2 = A^2 + AB + BA + B^2$$

Thus:

$$u_{exact}(\Delta t) = \left[ I + (A+B)\Delta t + \frac{1}{2}(A^2 + AB + BA + B^2) \Delta t^2 + \frac{1}{6}(A+B)^3 \Delta t^3 \right] u_0 + \mathcal{O}(\Delta t^4)$$



The Strang Splitting operator $S(\Delta t)$ is defined as:

$$S(\Delta t) = e^{A\frac{\Delta t}{2}} e^{B\Delta t} e^{A\frac{\Delta t}{2}}$$

Expanding each exponential term:

$$e^{A\frac{\Delta t}{2}} = I + \frac{1}{2}A\Delta t + \frac{1}{8}A^2 \Delta t^2 + \frac{1}{48}A^3 \Delta t^3 + \mathcal{O}(\Delta t^4)$$

$$e^{B\Delta t} = I + B\Delta t + \frac{1}{2}B^2 \Delta t^2 + \frac{1}{6}B^3 \Delta t^3 + \mathcal{O}(\Delta t^4)$$

Multiplying the operators:

$$S(\Delta t) = \left( I + \frac{1}{2}A\Delta t + \frac{1}{8}A^2 \Delta t^2 \right) \left( I + B\Delta t + \frac{1}{2}B^2 \Delta t^2 \right) \left( I + \frac{1}{2}A\Delta t + \frac{1}{8}A^2 \Delta t^2 \right) + \mathcal{O}(\Delta t^3)$$

$\Delta t^1$ term: $\frac{1}{2}A + B + \frac{1}{2}A = A + B$

$\Delta t^2$ term: $\frac{1}{8}A^2 + \frac{1}{2}B^2 + \frac{1}{8}A^2 + \frac{1}{2}AB + \frac{1}{2}BA + \frac{1}{4}A^2 = \frac{1}{2}(A^2 + B^2 + AB + BA) = \frac{1}{2}(A+B)^2$

#### Local Truncation Error

Since $S(\Delta t)$ matches the exact expansion up to the $\Delta t^2$ term:

$$\text{Local Truncation Error} = \| u_{exact}(\Delta t) - S(\Delta t)u_0 \| = \mathcal{O}(\Delta t^3)$$

#### Global Truncation Error

Let $T$ be the total simulation time, and $N = \frac{T}{\Delta t}$ be the number of steps. The global error $E$ is the accumulation of local errors:

$$E = \sum_{n=1}^{N} \text{Local Truncation Error}_n \approx N \cdot \mathcal{O}(\Delta t^3)$$

Substituting $N = \frac{T}{\Delta t}$:

$$E \approx \frac{T}{\Delta t} \cdot C \Delta t^3 = C T \Delta t^2$$

$$\text{,where C } = \left( \frac{1}{12}[B, [B, A]] + \frac{1}{24}[A, [A, B]] \right) u(t) , \quad\text{and} \quad [A,B] = AB - BA$$

Therefore, the global convergence rate is:

$$\text{Global Truncation Error} = \mathcal{O}(\Delta t^2)$$

<br></br>
<br></br>
<br></br>


#### b)
Spatial Truncation Error Proof for Central Difference

Proof:

Assume $u(x)$ is sufficiently smooth. Expanding $u$ about $x$ using Taylor's theorem:

$$u(x + \Delta x) = u(x) + \Delta x \frac{\partial u}{\partial x} + \frac{\Delta x^2}{2} \frac{\partial^2 u}{\partial x^2} + \frac{\Delta x^3}{6} \frac{\partial^3 u}{\partial x^3} + \frac{\Delta x^4}{24} \frac{\partial^4 u}{\partial x^4} + \mathcal{O}(\Delta x^5) \quad \text{--- (1)}$$

$$u(x - \Delta x) = u(x) - \Delta x \frac{\partial u}{\partial x} + \frac{\Delta x^2}{2} \frac{\partial^2 u}{\partial x^2} - \frac{\Delta x^3}{6} \frac{\partial^3 u}{\partial x^3} + \frac{\Delta x^4}{24} \frac{\partial^4 u}{\partial x^4} + \mathcal{O}(\Delta x^5) \quad \text{--- (2)}$$

Adding equations (1) and (2):

$$u(x + \Delta x) + u(x - \Delta x) = 2u(x) + \Delta x^2 \frac{\partial^2 u}{\partial x^2} + \frac{\Delta x^4}{12} \frac{\partial^4 u}{\partial x^4} + \mathcal{O}(\Delta x^6)$$


Rearranging the terms to solve for the second derivative $\frac{\partial^2 u}{\partial x^2}$:

$$\Delta x^2 \frac{\partial^2 u}{\partial x^2} = u(x + \Delta x) - 2u(x) + u(x - \Delta x) - \frac{\Delta x^4}{12} \frac{\partial^4 u}{\partial x^4} + \mathcal{O}(\Delta x^6)$$

Dividing by $\Delta x^2$:

$$\frac{\partial^2 u}{\partial x^2} = \frac{u(x + \Delta x) - 2u(x) + u(x - \Delta x)}{\Delta x^2} - \frac{\Delta x^2}{12} \frac{\partial^4 u}{\partial x^4} + \mathcal{O}(\Delta x^4)$$


The finite difference approximation is:

$$\left( \frac{\partial^2 u}{\partial x^2} \right)_{FDM} = \frac{u(x + \Delta x) - 2u(x) + u(x - \Delta x)}{\Delta x^2}$$

The truncation error $\text{Error}_{FDM}$ is the difference between the exact derivative and the numerical approximation:

$$\text{Error}_{FDM} = \left| \frac{\partial^2 u}{\partial x^2} - \left( \frac{\partial^2 u}{\partial x^2} \right)_{FDM} \right|$$

Substituting the derived expression:

$$\text{Error}_{FDM} = \frac{\Delta x^2}{12} \left| \frac{\partial^4 u}{\partial x^4} \right| + \mathcal{O}(\Delta x^4)$$

For the 2D Laplacian $\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$, the leading order error term is:

$$\text{Error}_{spatial} = \mathcal{O}\left( \Delta x^2 \cdot \frac{\partial^4 u}{\partial x^4} + \Delta y^2 \cdot \frac{\partial^4 u}{\partial y^4} \right)$$

<br></br>
<br></br>


#### Derivation of the Poisson Solver using Spectral Method (DST)

The Fisher-KPP equation is a reaction-diffusion equation:

$$\frac{\partial u}{\partial t} = D \nabla^2 u + \rho u \left(1 - \frac{u}{K}\right)$$

When we isolate the spatial operator and consider the steady-state or the pure diffusion limit without reaction and time dependence, we arrive at the Poisson Equation:

$$\nabla^2 u(x, y) = f(x, y)$$

where $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$ is the Laplacian operator. In the context of your research, solving this equation allows for the isolation of spatial discretization errors from temporal errors.


Given Dirichlet boundary conditions $u=0$ on the boundaries of a square domain $[0, L] \times [0, L]$, we expand $u(x, y)$ and $f(x, y)$ using the orthonormal basis functions $\phi_{m,n}(x, y) = \sin\left(\frac{m\pi x}{L}\right)\sin\left(\frac{n\pi y}{L}\right)$:

$$u(x, y) = \sum_{m=1}^{N} \sum_{n=1}^{N} \hat{u}_{m,n} \sin\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi y}{L}\right)$$

$$f(x, y) = \sum_{m=1}^{N} \sum_{n=1}^{N} \hat{f}_{m,n} \sin\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi y}{L}\right)$$

where $\hat{f}_{m,n}$ are the coefficients obtained via the Discrete Sine Transform (DST):

$$\hat{f}_{m,n} = \mathcal{DST}\{f(x, y)\}$$


Applying $\nabla^2$ to the expansion of $u(x, y)$:

$$\nabla^2 u = \sum_{m,n} \hat{u}_{m,n} \left[ \frac{\partial^2}{\partial x^2} \sin\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi y}{L}\right) + \frac{\partial^2}{\partial y^2} \sin\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi y}{L}\right) \right]$$

Using the property $\frac{d^2}{dx^2} \sin(kx) = -k^2 \sin(kx)$:

$$\nabla^2 u = \sum_{m,n} \hat{u}_{m,n} \left[ -\left(\frac{m\pi}{L}\right)^2 - \left(\frac{n\pi}{L}\right)^2 \right] \sin\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi y}{L}\right)$$

Substituting the expansions back into $\nabla^2 u = f$:

$$\sum_{m,n} \lambda_{m,n} \hat{u}_{m,n} \sin\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi y}{L}\right) = \sum_{m,n} \hat{f}_{m,n} \sin\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi y}{L}\right)$$

where the spectral eigenvalues $\lambda_{m,n}$ are:

$$\lambda_{m,n} = -\left[ \left(\frac{m\pi}{L}\right)^2 + \left(\frac{n\pi}{L}\right)^2 \right]$$

Due to the orthogonality of the sine basis, the coefficients must satisfy the algebraic relation:

$$\lambda_{m,n} \hat{u}_{m,n} = \hat{f}_{m,n}$$

Solving for the unknown coefficients $\hat{u}_{m,n}$:

$$\hat{u}_{m,n} = \frac{\hat{f}_{m,n}}{\lambda_{m,n}}$$

The final numerical solution $u(x, y)$ is obtained by the Inverse Discrete Sine Transform (IDST):

$$u(x, y) = \mathcal{IDST} \left\\{ \frac{\hat{f}_{m,n}}{\lambda_{m,n}} \right\\}$$
