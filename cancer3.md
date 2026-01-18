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
