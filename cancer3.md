要解下面的方程

$$
\frac{\partial u}{\partial t} = D \nabla^2 u + \rho u \left(1 - \frac{u}{K}\right) + f(x,y,t)
$$

為了求出對應的源項 $f$，將方程移項整理為：

$$
f(x,y,t) = \underbrace{\frac{\partial u}{\partial t}}_{\text{Term I}} - \underbrace{D \nabla^2 u}_{\text{Term II}} - \underbrace{\rho u \left(1 - \frac{u}{K}\right)}_{\text{Term III}}
$$

為了簡化，我們定義滿足 Homogeneous Dirichlet 邊界條件的空間基底函數 $\Phi(x,y)$：

$$
\Phi(x,y) = x(L-x)y(L-y)
$$

<br></br>

**情況一: $$v=1$$**

令 $$u$$ 不隨時間變化

$$
u(x,y,t) = \Phi(x,y) = x(L-x)y(L-y)
$$


由於 $$u$$ 與時間無關：

$$
\frac{\partial u}{\partial t} = 0
$$


先對 $x$ 偏微分兩次：

$$
\begin{aligned}
\frac{\partial}{\partial x} [x(L-x)] &= L - 2x \\
\frac{\partial^2}{\partial x^2} [x(L-x)] &= -2
\end{aligned}
$$

Hense

$$
\frac{\partial^2 u}{\partial x^2} = (-2) \cdot y(L-y)
$$

同理對 $y$ 偏微分：

$$
\frac{\partial^2 u}{\partial y^2} = x(L-x) \cdot (-2)
$$

合併得到 Laplacian：

$$
\nabla^2 u = -2 \left[ y(L-y) + x(L-x) \right]
$$

代入移項後的公式：

$$
\begin{aligned}
f(x,y,t) &= 0 - D \left[ -2 [y(L-y) + x(L-x)] \right] - \rho \Phi \left(1 - \frac{\Phi}{K}\right) \\
&= 2D [y(L-y) + x(L-x)] - \rho \Phi(x,y) \left(1 - \frac{\Phi(x,y)}{K}\right)
\end{aligned}
$$

<br></br>

**情況二: $$v=\sin(t)$$**

$$
u(x,y,t) = \Phi(x,y) \sin(t)
$$


對 $$t$$ 微分， $$\Phi(x,y)$$ 視為常數：

$$
\frac{\partial u}{\partial t} = \Phi(x,y) \frac{d}{dt}[\sin(t)] = \Phi(x,y) \cos(t)
$$


空間微分算子 $$\nabla^2$$ 不影響時間項 $$\sin(t)$$，直接引用情況一的結果：

$$
\begin{aligned}
\nabla^2 u &= \sin(t) \cdot \nabla^2 [\Phi(x,y)] \\
&= \sin(t) \cdot \{-2 [y(L-y) + x(L-x)]\}
\end{aligned}
$$


直接代入 $u$：

$$
\text{Reaction} = \rho [\Phi \sin(t)] \left( 1 - \frac{\Phi \sin(t)}{K} \right)
$$


將上述三項代入公式 $$f = \text{Term I} - D(\text{Term II}) - \text{Term III}$$：

$$
\begin{aligned}
f(x,y,t) &= \Phi \cos(t) \\
&\quad - D \left[ -2 \sin(t) [y(L-y) + x(L-x)] \right] \\
&\quad - \rho \Phi \sin(t) \left( 1 - \frac{\Phi \sin(t)}{K} \right)
\end{aligned}
$$

最終整理

$$
\begin{aligned}
f(x,y,t) &= \Phi(x,y)\cos(t) \\
&\quad + 2D\sin(t) \left[ y(L-y) + x(L-x) \right] \\
&\quad - \rho \Phi(x,y)\sin(t) + \frac{\rho}{K} [\Phi(x,y)\sin(t)]^2
\end{aligned}
$$


<br></br>


**情況三: $$v = 2 + \sin(\frac{4\pi x}{L} - 3t)$$。**

定義:

$$
\frac{\partial u}{\partial t} = D \nabla^2 u + \rho u \left(1 - \frac{u}{K}\right) + f(x,y,t)
$$

移項求 $$f$$：

$$
f(x,y,t) = \frac{\partial u}{\partial t} - D \nabla^2 u - \rho u \left(1 - \frac{u}{K}\right)
$$


$$
v(x,y,t) = 2 + \sin(\theta), \quad \text{where } \theta = \frac{4\pi x}{L} - 3t
$$

(註：常數 $$2$$ 用於保持數值為正； $$4\pi$$ 代表在長度 $$L$$ 內有兩個波峰； $$3t$$ 控制波速)

$$
u(x,y,t) = \Phi(x,y) \cdot v(x,y,t)
$$


$$
\frac{\partial v}{\partial t} = \cos(\theta) \cdot \frac{\partial \theta}{\partial t} = -3 \cos(\theta)
$$

因此：

$$
\frac{\partial u}{\partial t} = \Phi \cdot \frac{\partial v}{\partial t} = -3 \Phi(x,y) \cos\left(\frac{4\pi x}{L} - 3t\right)
$$


利用恆等式 $$\nabla^2 (\Phi v) = v \nabla^2 \Phi + 2 \nabla \Phi \cdot \nabla v + \Phi \nabla^2 v$$


由於 $$v$$ 只與 $$x$$ 有關：

$$
\begin{aligned}
\frac{\partial v}{\partial x} &= \frac{4\pi}{L} \cos(\theta) \\
\frac{\partial^2 v}{\partial x^2} &= -\left(\frac{4\pi}{L}\right)^2 \sin(\theta)
\end{aligned}
$$

所以：

$$
\nabla^2 v = -\left(\frac{4\pi}{L}\right)^2 \sin(\theta)
$$


已知：

$$
\begin{aligned}
\nabla^2 \Phi &= -2 [y(L-y) + x(L-x)] \\
\frac{\partial \Phi}{\partial x} &= (L-2x)y(L-y)
\end{aligned}
$$


$$
\begin{aligned}
\nabla^2 u &= v \cdot \nabla^2 \Phi \\
&\quad + 2 \left[ \frac{\partial \Phi}{\partial x} \frac{\partial v}{\partial x} \right] \\
&\quad + \Phi \cdot \nabla^2 v
\end{aligned}
$$

代入具體函數：

$$
\begin{aligned}
\nabla^2 u &= \left[2+\sin(\theta)\right] \cdot \{-2 [y(L-y) + x(L-x)]\} \\
&\quad + 2 \left[ (L-2x)y(L-y) \cdot \frac{4\pi}{L} \cos(\theta) \right] \\
&\quad + \Phi \cdot \left[ -\left(\frac{4\pi}{L}\right)^2 \sin(\theta) \right]
\end{aligned}
$$

將上述結果代入 $$f = u_t - D\nabla^2 u - \text{Reaction}$$

$$
\begin{aligned}
f(x,y,t) &= -3 \Phi \cos(\theta) \\
&\quad - D \left[ v (\nabla^2 \Phi) + \frac{8\pi}{L} (L-2x)y(L-y) \cos(\theta) - \left(\frac{4\pi}{L}\right)^2 \Phi \sin(\theta) \right] \\
&\quad - \rho u \left( 1 - \frac{u}{K} \right)
\end{aligned}
$$

其中 $$\theta = \frac{4\pi x}{L} - 3t$$

<br></br>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dst, idst
import time

class FisherKPPMMS_DST:
    def __init__(self, N=64, L=50.0, D=1.0, rho=1.0, K=100.0):
        self.N = N
        self.L = L
        self.D = D
        self.rho = rho
        self.K = K
        self.dx = L / (N + 1)
        
        # 建立網格
        x = np.linspace(self.dx, L - self.dx, N)
        y = np.linspace(self.dx, L - self.dx, N)
        self.X, self.Y = np.meshgrid(x, y)
        

        self.scale = 1e-4 

        # --- 預計算特徵值 ---
        k = np.arange(1, N + 1)
        kx, ky = np.meshgrid(k, k)
        
        # 1. Spectral Eigenvalues
        self.lambda_spec = -((kx * np.pi / L)**2 + (ky * np.pi / L)**2)
        
        # 2. FDM Eigenvalues (via DST)
        lx_fdm = (2 / self.dx**2) * (np.cos(kx * np.pi / (N + 1)) - 1)
        ly_fdm = (2 / self.dx**2) * (np.cos(ky * np.pi / (N + 1)) - 1)
        self.lambda_fdm = lx_fdm + ly_fdm

        # 預計算空間基底 Phi 
        # Phi = scale * x(L-x)y(L-y)
        raw_Phi = self.X * (self.L - self.X) * self.Y * (self.L - self.Y)
        self.Phi = self.scale * raw_Phi
        raw_neg_lap = 2 * (self.Y * (self.L - self.Y) + self.X * (self.L - self.X))
        self.neg_lap_Phi = self.scale * raw_neg_lap

    def get_mms_data(self, t, case):
        """
        MMS Exact Solution and Source Term Generator
        """
        Phi = self.Phi
        neg_lap_Phi = self.neg_lap_Phi
        
        if case == 1:
            # Case 1: Static (v=1)
            u_exact = Phi
            reaction = self.rho * u_exact * (1 - u_exact / self.K)
            f_source = self.D * neg_lap_Phi - reaction

        elif case == 2:
            # Case 2: Dynamic (v=sin(t))
            v = np.sin(t)
            dv_dt = np.cos(t)
            u_exact = Phi * v
            
            term_dt = Phi * dv_dt
            term_diff = self.D * v * neg_lap_Phi 
            reaction = self.rho * u_exact * (1 - u_exact / self.K)
            
            f_source = term_dt + term_diff - reaction

        elif case == 3:
            # Case 3: Traveling Wave
            k_wave = 4 * np.pi / self.L
            omega = 3.0
            theta = k_wave * self.X - omega * t
            
            v = 2 + np.sin(theta)
            u_exact = Phi * v
            
            du_dt = -omega * Phi * np.cos(theta)
            
            term_A = v * (-neg_lap_Phi)
            
            dPhi_dx = self.scale * (self.L - 2 * self.X) * self.Y * (self.L - self.Y)
            dv_dx = k_wave * np.cos(theta)
            term_B = 2 * dPhi_dx * dv_dx
            
            lap_v = -(k_wave**2) * np.sin(theta)
            term_C = Phi * lap_v
            
            lap_u = term_A + term_B + term_C
            term_diff = -self.D * lap_u
            
            reaction = self.rho * u_exact * (1 - u_exact / self.K)
            
            f_source = du_dt + term_diff - reaction
            
        return u_exact, f_source

    def reaction_exact(self, u, dt):
        u = np.maximum(u, 0)
        exp_rho = np.exp(self.rho * dt)
        numerator = self.K * u * exp_rho
        denominator = self.K + u * (exp_rho - 1)
        return numerator / denominator

    def solve(self, u0, T, dt, case, method='spectral'):
        u = u0.copy()
        steps = int(T / dt)
        t = 0.0
        
        if method == 'spectral':
            lambdas = self.lambda_spec
        elif method == 'fdm':
            lambdas = self.lambda_fdm
        else:
            raise ValueError("Method must be 'spectral' or 'fdm'")
            
        decay = np.exp(self.D * lambdas * dt)
        
        for _ in range(steps):
            # 1. Reaction (dt/2)
            u = self.reaction_exact(u, dt/2)
            
            # 2. Diffusion + Source (dt)
            _, f_source = self.get_mms_data(t, case) # Source at t
            
            # DST Transform
            u_hat = dst(dst(u, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
            f_hat = dst(dst(f_source, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
            
            # Exact Integration
            with np.errstate(divide='ignore', invalid='ignore'):
                factor = (decay - 1) / (self.D * lambdas)
                factor = np.where(np.abs(lambdas) < 1e-12, dt, factor)
                
            u_hat_new = u_hat * decay + f_hat * factor
            
            # IDST
            u = idst(idst(u_hat_new, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
            
            t += dt
            
            # 3. Reaction (dt/2)
            u = self.reaction_exact(u, dt/2)
            
        return u

def run_simulation():
    # Setup
    N = 64
    L = 50.0
    T = 1.0  
    dt = 1e-5
    
    solver = FisherKPPMMS_DST(N=N, L=L)
    
    cases = [
        (1, "Case 1: Static (v=1)"),
        (2, "Case 2: Dynamic (v=sin(t))"),
        (3, "Case 3: Traveling Wave")
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    for row, (case_id, title) in enumerate(cases):
        print(f"Simulating {title}...")
        
        u0, _ = solver.get_mms_data(0, case_id)
        u_exact, _ = solver.get_mms_data(T, case_id)
        
        # Solving
        u_fdm = solver.solve(u0, T, dt, case_id, method='fdm')
        u_spec = solver.solve(u0, T, dt, case_id, method='spectral')
        
        # Errors (Absolute Difference)
        err_fdm = np.abs(u_exact - u_fdm)
        err_spec = np.abs(u_exact - u_spec)
        
        # 1. Exact Solution
        # 使用 scale 後數值變小，colorbar 會自動調整
        im0 = axes[row, 0].imshow(u_exact, origin='lower', extent=[0,L,0,L], cmap='inferno')
        axes[row, 0].set_title(f"{title}\nExact Solution (t={T})")
        plt.colorbar(im0, ax=axes[row, 0])
        
        # 2. FDM Error
        im1 = axes[row, 1].imshow(err_fdm, origin='lower', extent=[0,L,0,L], cmap='jet')
        axes[row, 1].set_title(f"FDM Error (via DST)\nMax: {np.max(err_fdm):.2e}")
        plt.colorbar(im1, ax=axes[row, 1])
        
        # 3. Spectral Error
        im2 = axes[row, 2].imshow(err_spec, origin='lower', extent=[0,L,0,L], cmap='jet')
        axes[row, 2].set_title(f"Spectral Error\nMax: {np.max(err_spec):.2e}")
        plt.colorbar(im2, ax=axes[row, 2])
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()
```
![figure](2.jpg)

![figure](3.jpg)

![figure](4.jpg)
