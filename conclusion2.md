* Step 1: 探討 $u \to \Delta u$（空間算子的本質)
  * 這一步專注於探討空間離散化最基礎的「曲率」計算，並利用特徵值揪出 FDM 在高頻波段的誤差源頭。
* Step 2: $f = \Delta u \implies u = \Delta^{-1} f$（求反矩陣 / 空間特徵值除法
  * 這是一個靜態系統。透過拔除時間變數 $t$，把空間誤差徹底孤立出來，為 AI 創造出 $100\%$ 乾淨的空間修正訓練數據。
* Step 3: $u_t = \Delta u$（純擴散動態）
  * 這是一個純線性動態系統。加入時間演化，但利用頻域的「精確解析解」凍結時間誤差，藉此觀察純空間誤差如何在動態過程中引發「散熱太慢（數值遲滯）」的現象。
* Step 4: $u_t = D\Delta u + \rho u(1-\frac{u}{K})$（最終非線性系統）
  * 證明當反應與擴散耦合時，時間步進誤差與空間誤差會在「陡峭波前」產生致命的複合爆炸（對易子誤差），從而突顯出我們為什麼必須依賴前三步來幫 AI 進行「前置去耦訓練」。


<br></br>

#### Step 1: Laplacian 算子
<br></br>
為了解決「為什麼 FDM 在高頻會產生龐大誤差」這個核心痛點。這個範例直接將「泰勒展開式」、「FDM 差分格式」與「矩陣特徵值」這三個原本看似獨立的概念完美串接在一起。

##### Laplacian 算子 —— 空間離散化之基礎與誤差發源地

作用：定義空間網格上物理量分布的「凹凸程度（曲率）」，這是推動所有熱傳導與擴散行為的數學基石，同時也是有限差分法（FDM）系統性誤差的唯一發源地。



###### $$\color{red}{連續空間的定義}$$

在連續的二維笛卡爾座標系中，Laplacian 算子定義為純粹的二次偏微分，它精確衡量了函數在空間中的極限變化率：

$$\nabla^2 u(x,y) = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$$

###### $$\color{red}{有限差分法 (FDM) 的空間離散化與誤差展開}$$
在實際數值計算中，我們必須在離散網格上（間距為 $\Delta x, \Delta y$）進行逼近。我們利用泰勒展開式 (Taylor Series) 來展開相鄰網格點的數值。
對於 $$x$$ 方向的中心差分 (Central Difference)：

$$u(x+\Delta x) = u(x) + \Delta x \frac{\partial u}{\partial x} + \frac{\Delta x^2}{2} \frac{\partial^2 u}{\partial x^2} + \frac{\Delta x^3}{6} \frac{\partial^3 u}{\partial x^3} + \frac{\Delta x^4}{24} \frac{\partial^4 u}{\partial x^4} + \mathcal{O}(\Delta x^5)$$

$$u(x-\Delta x) = u(x) - \Delta x \frac{\partial u}{\partial x} + \frac{\Delta x^2}{2} \frac{\partial^2 u}{\partial x^2} - \frac{\Delta x^3}{6} \frac{\partial^3 u}{\partial x^3} + \frac{\Delta x^4}{24} \frac{\partial^4 u}{\partial x^4} + \mathcal{O}(\Delta x^5)$$

將上述兩式相加並重新排列，我們得到 FDM 二階中心差分的運算子定義，及其伴隨的截斷誤差 (Truncation Error)：

$$\frac{u(x+\Delta x) - 2u(x) + u(x-\Delta x)}{\Delta x^2} = \frac{\partial^2 u}{\partial x^2} + \underbrace{\frac{\Delta x^2}{12} \frac{\partial^4 u}{\partial x^4}}_{\text{主導誤差項 (Leading Error)}} + \mathcal{O}(\Delta x^4)$$

移項後可得：

$$\text{真實微分 } \left(\frac{\partial^2 u}{\partial x^2}\right) = \left(\frac{\partial^2 u}{\partial x^2}\right)_{FDM} - \frac{\Delta x^2}{12} \frac{\partial^4 u}{\partial x^4} - \mathcal{O}(\Delta x^4)$$

結論：

FDM 的空間誤差精度為 $$\mathcal{O}(\Delta x^2)$$，且該誤差量值直接正比於波形的四階微分（ $$\frac{\partial^4 u}{\partial x^4}$$ ）。這意味著當波形震盪越劇烈（高頻），其四階導數會呈幾何級數放大，導致 FDM 算子產生毀滅性的空間扭曲。


###### $$\color{red}{頻譜法 (Spectral Method) 的空間離散化與誤差展開}$$

與 FDM 依賴「局部相鄰網格點」來逼近微積分不同，頻譜法採用「全域正交基底函數 (Global Orthogonal Basis Functions)」的無窮級數來離散化空間。在 Dirichlet 邊界條件下，使用傅立葉正弦級數 (Fourier Sine Series) 來展開真實的空間函數 $$u(x)$$ ：

$$u(x) = \sum_{m=1}^{\infty} \hat{u}_m \sin(k_m x), \quad k_m = \frac{m\pi}{L}$$

where $$\hat{u}_m:$$ (Fourier Sine Coefficient) 
是透過將原函數 $$u(x)$$ 投影 (Projection) 到正交基底 $$\sin(k_m x)$$ 上所計算出來的。

利用三角函數的正交性（Orthogonality），等式兩邊同乘 $$\sin(k_p x)$$ 並在空間 $$[0, L]$$ 上積分，藉此嚴格求出 $$\hat{u}_m$$ 的公式： 

$$\hat{u}_m = \frac{2}{L} \int_{0}^{L} u(x) \sin(k_m x) dx$$

在實際數值計算中，我們無法計算到無窮大，必須在特定的波數 $N$（對應網格解析度）進行截斷 (Truncation)，得到數值逼近解 $$u_N(x)$$ ：

$$u_N(x) = \sum_{m=1}^{N} \hat{u}_m \sin(k_m x)$$

空間導數的精確性：
與 FDM 使用差分逼近不同，頻譜法的空間導數是直接在頻域中進行「解析微分 (Analytical Differentiation)」。對 $u_N(x)$ 取二次偏微分，微積分算子會直接作用在基底上，毫無逼近妥協：

$$\frac{\partial^2 u_N}{\partial x^2} = \sum_{m=1}^{N} \hat{u}_m \left(-k_m^2\right) \sin(k_m x)$$

截斷誤差 (Truncation Error) 分析：
頻譜法的唯一空間誤差來源，是那些被我們捨棄的高頻模態（波數 $$m > N$$ 的部分）。其空間截斷誤差 $$E_N(x)$$ 嚴格定義為真實解與數值解的差值：

$$E_N(x) = u(x) - u_N(x) = \sum_{m=N+1}^{\infty} \hat{u}_m \sin(k_m x)$$

根據傅立葉分析的達布定理 (Darboux's Principle)，頻譜係數 $$\hat{u}_m$$ 的衰減速度，完全取決於真實函數 $$u(x)$$ 的平滑程度（可微性）：
1. 若函數只有 $p$ 階連續可微 ( $$u \in C^p$$ )，則係數以代數速率衰減： $$|\hat{u}_m| \sim \mathcal{O}(m^{-(p+1)})$$ 。
2. 頻譜收斂 (Spectral Convergence)：若函數為無限階可微的平滑函數 ( $$u \in C^\infty$$ ，例如高斯波或三角函數)，其頻譜係數將以超越任何多項式的速度下降，呈現指數級衰減：
   $$|\hat{u}_m| \sim \mathcal{O}(e^{-cm}), \quad c > 0$$   ( $$c$$ 是函數在複數平面上的「解析帶寬 (Width of Analytic Strip)」，也就是從實數軸出發，抵達最近一個奇異點的最短距離。)


>達布定理最初是針對泰勒級數 (Taylor Series) 或冪級數提出的，後來被廣泛推廣到傅立葉級數與切比雪夫級數。它的核心數學表述如下：
>Darboux's Theorem :
>
>假設一個複變函數 $f(z)$ 在原點的某個鄰域內是解析的 (Analytic)，並具有冪級數展開： $$f(z) = \sum_{n=0}^{\infty} a_n z^n$$ 若  $$f(z)$$ 的收斂半徑為 $R$，且在收斂圓 $$|z| = R$$ 上存在有限個奇異點 (Singularities) $$\zeta_k$$ 。那麼，當 $$n \to \infty$$ 時，級數係數 $$a_n$$ 的漸近行為 
>(Asymptotic behavior)，完全由函數 $$f(z)$$ 在這些邊界奇異點 $$\zeta_k$$ 附近的局部行為所決定。

將此性質代回誤差方程式，我們可得出頻譜法的全域空間誤差上界：

$$\|E_N\| \leq C e^{-cN}$$

>它是複變分析 (Complex Analysis) 結合傅立葉分析 (Fourier Analysis)所推導出來的嚴格數學結果。
>
>##### 第一步：定義空間截斷誤差 (Truncation Error)
>
>假設我們有一個在區間 $[-\pi, \pi]$ 上的週期函數 $u(x)$，它的完整傅立葉級數展開為：
>$$u(x) = \sum_{m=-\infty}^{\infty} \hat{u}_m e^{imx}$$
>
>在數值計算中，只能保留前 $N$ 個波數（頻率）時，我們的數值近似解為：
>$$u_N(x) = \sum_{m=-N}^{N} \hat{u}_m e^{imx}$$
>
>那麼，空間截斷誤差 $E_N(x)$ 就是那些被我們捨棄的「高頻尾巴 (High-frequency tail)」：
>$$E_N(x) = u(x) - u_N(x) = \sum_{|m| > N} \hat{u}_m e^{imx}$$
>
>我們想要知道這個誤差的最大值（即無窮範數 $\|E_N\|_\infty$），利用三角不等式可以得到誤差的上界：
>
>$$\|E_N\| \leq \sum_{|m| > N} |\hat{u}_m|$$
>
>>
>>$$E_N(x) = \sum_{|m| > N} \hat{u}_m e^{imx}$$
>>
>>公式左邊的 $$\|E_N\|$$ 加上了雙直線，在數學上稱為 (Norm)。在這裡，它通常代表的是最大絕對值範數 (Infinity Norm, $$\| \cdot \|_\infty$$ )。
>>
>>$$|a + b| \leq |a| + |b|$$
>>
>>把這個概念推廣到無窮多項的連加（Sigma $\sum$），就是廣義的三角不等式:
>>
>>$$\left| \sum a_k \right| \leq \sum |a_k|$$
>>
>>現在，我們對誤差函數 $E_N(x)$ 的兩邊同時取絕對值，並套用廣義三角不等式：
>>
>>$$|E_N(x)| = \left| \sum_{|m| > N} \hat{u}_m e^{imx} \right| \leq \sum_{|m| > N} \left| \hat{u}_m e^{imx} \right|$$
>>
>>
>>接下來是這個推導中最關鍵的一步：處理括號裡面的 $\left| \hat{u}_m e^{imx} \right|$。
>>
>>根據絕對值的乘法性質 $|A \cdot B| = |A| \cdot |B|$，我們可以把它拆開：
>>
>>$$\left| \hat{u}_m e^{imx} \right| = |\hat{u}_m| \cdot |e^{imx}|$$
>>
>>根據複變函數中的尤拉公式 (Euler's Formula)：
>>
>>$$e^{imx} = \cos(mx) + i \sin(mx)$$
>>
>>它的絕對值（長度）為實部平方加虛部平方再開根號：
>>
>>$$|e^{imx}| = \sqrt{\cos^2(mx) + \sin^2(mx)} = \sqrt{1} = 1$$
>>
>>這代表在複數平面上， $$e^{imx}$$ 永遠是在半徑為 $$1$$ 的單位圓上旋轉，它只會改變相位（角度），完全不會改變長度（振幅）。
>>
>>因此，把 $|e^{imx}| = 1$ 代回去：
>>
>>$$|E_N(x)| \leq \sum_{|m| > N} |\hat{u}_m| \cdot 1$$
>>
>>$$|E_N(x)| \leq \sum_{|m| > N} |\hat{u}_m|$$
>>
>
>
>##### 第二步：利用柯西積分定理求 $|\hat{u}_m|$ 的衰減率 (The Contour Shift)
>
>傅立葉係數的原始定義為實數軸上的積分：
>
>$$\hat{u}_m = \frac{1}{2\pi} \int_{-\pi}^{\pi} u(x) e^{-imx} dx$$
>
>現在，把實數變數 $x$ 擴展為複數變數 $$z = x + iy$$ 。
>假設 $u(z)$ 在以實數軸為中心、寬度為 $c$ 的水平帶狀區域內（$-c < \text{Im}(z) < c$）是*解析的 (Analytic)，這代表在這個帶狀區域內沒有任何奇異點，且函數值是有界的，我們設其最大值為 $M$（即 $|u(z)| \leq M$）。
>
>根據複變函數中的柯西積分定理 (Cauchy's Integral Theorem)，如果在一個解析區域內平移積分路徑，積分的結果不會改變。
>
>* 當 $m > 0$ 時，我們將積分路徑從實數軸向下平移到 $z = x - ic$：
>
>  $$\hat{u}_m = \frac{1}{2\pi} \int_{-\pi}^{\pi} u(x - ic) e^{-im(x - ic)} dx$$
>
>把指數項拆開 $e^{-im(x - ic)} = e^{-imx} e^{-m c}$：
>
>$$\hat{u}_m = \frac{e^{-mc}}{2\pi} \int_{-\pi}^{\pi} u(x - ic) e^{-imx} dx$$
>
>
>現在，對等式兩邊取絕對值來尋找上界：
>
>$$|\hat{u}_m| \leq \frac{e^{-mc}}{2\pi} \int_{-\pi}^{\pi} |u(x - ic)| \cdot |e^{-imx}| dx$$
>
>因為 $|e^{-imx}| = 1$，且我們假設在該路徑上 $|u(z)| \leq M$：
>
>$$|\hat{u}_m| \leq \frac{e^{-mc}}{2\pi} \int_{-\pi}^{\pi} M dx = \frac{e^{-mc}}{2\pi} (2\pi M) = M e^{-mc}$$
>
>同理，當 $m < 0$ 時，我們將積分路徑向上平移到 $z = x + ic$，可以得到完全對稱的結果 $|\hat{u}_m| \leq M e^{-|m|c}$。
>
>結論： 透過複數平面的路徑平移，我們嚴格證明了單一頻譜係數的衰減率為：
>
>$$|\hat{u}_m| \leq M e^{-|m|c}$$
>
>##### 第三步：對高頻尾巴進行等比級數求和 (Summing the Geometric Series)
>
>現在，把第二步求出的係數衰減率代回第一步的誤差公式中。
>
>空間截斷誤差的總和為：
>
>$$\|E_N\| \leq \sum_{|m| > N} |\hat{u}_m| = 2 \sum_{m = N+1}^{\infty} |\hat{u}_m|$$
>
>將 $|\hat{u}_m| \leq M e^{-mc}$ 代入：
>
>$$\|E_N\| \leq 2 \sum_{m = N+1}^{\infty} M e^{-mc} = 2M \sum_{m = N+1}^{\infty} (e^{-c})^m$$
>
>你會發現，這是一個公比為 $r = e^{-c}$ 的無窮等比級數 (Geometric Series)。因為 $$c > 0$$ ，所以 $$e^{-c} < 1$$ ，這個級數必然收斂。
>利用無窮等比級數求和公式 $$\sum_{k=K}^\infty r^k = \frac{r^K}{1-r}$$ ：
>
>$$\|E_N\| \leq 2M \frac{e^{-c(N+1)}}{1 - e^{-c}}$$
>
>$$\|E_N\| \leq \left( \frac{2M e^{-c}}{1 - e^{-c}} \right) \cdot e^{-cN}$$
>
>令常數 $$C = \frac{2M e^{-c}}{1 - e^{-c}}$$ ，就完美得到了最終的誤差上界定理：
>
>$$\|E_N\| \leq C e^{-cN}$$

結論：
頻譜法的空間誤差精度為 指數級收斂 $$\mathcal{O}(e^{-cN})$$ 。這意味著，只要物理場足夠平滑，當我們稍微增加網格點 $N$ 時，誤差就會像雪崩一樣迅速掉到機器的浮點數極限（ $$\approx 10^{-16}$$ ）。這在數學上完美解釋了為何在 Step 1 的後續推導中，Spectral Method 的特徵值 $$\lambda^{spec}$$ 可以被視為「絕對精確的 Ground Truth」，而 FDM 卻只能在 $$\mathcal{O}(\Delta x^2)$$ 的泥沼中掙扎。



##### $$\color{red}{空間算子的特徵值分析與三角函數範例}$$

為了嚴謹量化上述誤差是如何影響整體 PDE 系統的，我們將 $$\nabla^2$$ 算子作用於標準的傅立葉基底函數上進行檢驗。
考慮一個單一頻率的正弦波（其中波數 $$k_x = \frac{m\pi}{L}, k_y = \frac{n\pi}{L}$$ ）：

$$u(x,y) = \sin(k_x x) \sin(k_y y)$$

* A. 頻譜法 Spectral Method 的精確響應
將此函數代入連續的微積分算子中，直接求二次導函數：

$$\nabla^2 u(x,y) = \frac{\partial^2}{\partial x^2} [\sin(k_x x)\sin(k_y y)] + \frac{\partial^2}{\partial y^2} [\sin(k_x x)\sin(k_y y)]$$

$$\nabla^2 u(x,y) = -(k_x^2 + k_y^2) \cdot \sin(k_x x)\sin(k_y y)$$

由此可知，精確的頻譜特徵值（Eigenvalue）為：

$$\lambda_{m,n}^{spec} = -\left[ \left(\frac{m\pi}{L}\right)^2 + \left(\frac{n\pi}{L}\right)^2 \right]$$

* B. FDM 的離散響應與三角恆等式推導
現在，我們將同一個正弦波 $$u(x,y)$$ 代入 FDM 的差分格式中。僅觀察 $$x$$ 方向：

$$u(x+\Delta x) + u(x-\Delta x) = \sin(k_x(x+\Delta x))\sin(k_y y) + \sin(k_x(x-\Delta x))\sin(k_y y)$$

利用三角函數和差化積恆等式 $$\sin(A+B) + \sin(A-B) = 2\sin(A)\cos(B)$$ ，上式可化簡為：

$$= [2\sin(k_x x)\cos(k_x \Delta x)] \sin(k_y y)$$

將此結果代回 FDM 差分公式 $$\frac{u(x+\Delta x) - 2u(x) + u(x-\Delta x)}{\Delta x^2}$$ 中：

$$= \frac{2\sin(k_x x)\cos(k_x \Delta x) - 2\sin(k_x x)}{\Delta x^2} \sin(k_y y)$$

提取公因式 $\sin(k_x x)$：

$$= \left[ \frac{2}{\Delta x^2} (\cos(k_x \Delta x) - 1) \right] \cdot \underbrace{\sin(k_x x)\sin(k_y y)}_{u(x,y)}$$

擴展至二維空間，我們得到 FDM 的離散特徵值為：

$$\lambda_{m,n}^{fdm} = \frac{2}{\Delta x^2}\left(\cos\frac{m\pi\Delta x}{L} - 1\right) + \frac{2}{\Delta y^2}\left(\cos\frac{n\pi\Delta y}{L} - 1\right)$$

* C. 理論與近似的對接證明
利用 $\cos(\theta)$ 的泰勒展開式 $\cos(\theta) \approx 1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!}$，我們將 $\lambda_{m,n}^{fdm}$ 的 $x$ 分量展開：

$$\frac{2}{\Delta x^2} \left[ \left( 1 - \frac{(k_x \Delta x)^2}{2} + \frac{(k_x \Delta x)^4}{24} - \dots \right) - 1 \right]$$

$$= \frac{2}{\Delta x^2} \left[ -\frac{k_x^2 \Delta x^2}{2} + \frac{k_x^4 \Delta x^4}{24} \right] = -k_x^2 + \frac{\Delta x^2}{12}k_x^4$$

此式完美吻合了有限差分法 (FDM) 的空間離散化與誤差展開的理論推導： $$-k_x^2$$ 對應了精確的二次微分 $\lambda^{spec}$，而 $$\frac{\Delta x^2}{12}k_x^4$$ 正是對應了主導誤差項 $$\frac{\Delta x^2}{12} \frac{\partial^4 u}{\partial x^4}$$ ！

##### $$\color{blue}{step1 的結論}$$
透過上述嚴謹的代數證明，確立了 FDM 空間截斷誤差的數學本質：「FDM 演算法本質上高估了高頻波段的平滑度，導致物理耗散率的計算偏離真實值」。
在傳統數值分析中，這被視為無法跨越的網格限制（必須依靠加密網格 $$\Delta x \to 0$$ 來解決）。然而，在本研究中，這段從 $$\lambda_{fdm}$$ 映射回 $$\lambda_{spec}$$ 的解析差距。

link:https://colab.research.google.com/drive/15TOTS_xZIY6lpomE8A8ijzpu-E3o-zYn?usp=sharing

<br></br>
<br></br>
<br></br>

#### Step 2: Poisson 方程 —— 空間誤差完全去耦 (Static Decoupling)
<br></br>
作用：Poisson 方程是一個完全沒有時間變數 $t$ 的靜態邊值問題 (Boundary Value Problem)。藉由求解此方程，我們能將「時間截斷誤差」徹底剔除，萃取出 $100\%$ 純粹由 $\lambda_{m,n}^{fdm}$ 缺陷所導致的空間扭曲。這是建立神經網路乾淨訓練數據 (Ground Truth) 的最完美環境。

##### $$\color{red}{數學定義與物理直覺}$$

給定靜態方程式：

$$\nabla^2 u(x,y) = f(x,y)$$

* 物理直覺：想像一張固定在四面邊界的彈性薄膜。右側的 $$f(x,y)$$ 代表在空間各點施加的「靜態外力分佈」，而左側的 Laplacian 算子 $$\nabla^2$$ 代表薄膜本身的「彈性恢復力」。方程式求解的 $$u(x,y)$$ ，就是薄膜在受力平衡後最終的「靜態變形位移」。

##### $$\color{red}{頻域展開與正交性解耦推導 (Orthogonal Decoupling)}$$

為了解決這個偏微分方程，我們將未知變形 $$u(x,y)$$ 與已知受力 $$f(x,y)$$ 皆利用二維傅立葉正弦級數 (對應離散正弦轉換 DST) 展開：

$$u(x,y) = \sum_{m=1}^{N}\sum_{n=1}^{N} \hat{u}_{m,n} \sin\left(\frac{m\pi x}{L}\right)\sin\left(\frac{n\pi y}{L}\right)$$

$$f(x,y) = \sum_{m=1}^{N}\sum_{n=1}^{N} \hat{f}_{m,n} \sin\left(\frac{m\pi x}{L}\right)\sin\left(\frac{n\pi y}{L}\right)$$

推導步驟 1：算子代入

將展開式代入原微分方程 $$\nabla^2 u = f$$ 。因為 $$\nabla^2$$ 是線性算子，可以直接穿透總和符號作用在基底上：

$$\sum_{m=1}^{N}\sum_{n=1}^{N} \hat{u}_{m,n} \left[ \nabla^2 \left( \sin\frac{m\pi x}{L}\sin\frac{n\pi y}{L} \right) \right] = \sum_{m=1}^{N}\sum_{n=1}^{N} \hat{f}_{m,n} \sin\left(\frac{m\pi x}{L}\right)\sin\left(\frac{n\pi y}{L}\right)$$

根據 Step 1， $$\nabla^2$$ 作用在正弦基底上會產生特徵值 $$\lambda_{m,n}$$ ：

$$\sum_{m=1}^{N}\sum_{n=1}^{N} \hat{u}_{m,n} \cdot \lambda_{m,n} \cdot \sin\left(\frac{m\pi x}{L}\right)\sin\left(\frac{n\pi y}{L}\right) = \sum_{m=1}^{N}\sum_{n=1}^{N} \hat{f}_{m,n} \sin\left(\frac{m\pi x}{L}\right)\sin\left(\frac{n\pi y}{L}\right)$$

推導步驟 2：利用正交性 (Orthogonality) 萃取係數

正弦函數家族在給定區間內具備正交性。我們將等式兩側同乘上特定的基底 $$\sin(\frac{p\pi x}{L})\sin(\frac{q\pi y}{L})$$ 並對整個空間面積分：

$$\int_0^L \int_0^L (\text{左式}) \, dxdy = \int_0^L \int_0^L (\text{右式}) \, dxdy$$

因為正交性，當 $$(m,n) \neq (p,q)$$ 時，積分結果為 $$0$$ 。所有的交叉項皆被消滅，偏微分方程瞬間退化為 $$N \times N$$ 個獨立的代數方程：

$$\lambda_{m,n} \cdot \hat{u}_{m,n} = \hat{f}_{m,n}$$

透過簡單的代數除法（逆運算），即可求得頻域中的變形係數：

$$\hat{u}_{m,n} = \frac{\hat{f}_{m,n}}{\lambda_{m,n}}$$

##### $$\color{red}{數學範例：單一高頻源項的誤差放大證明}$$

為了精確證明 FDM 在 Poisson 方程中是如何產生誤差放大的，我們考慮一個極端的單一高頻外力： $$f(x,y) = \sin(k_x x)\sin(k_y y)$$ 。此時頻域中只有該頻率的 $$\hat{f}_{m,n} = 1$$ 。

* 頻譜法 Spectral Method 的精確變形

$$\hat{u}_{m,n}^{spec} = \frac{1}{\lambda_{m,n}^{spec}} = \frac{1}{-(k_x^2 + k_y^2)}$$

* 有限差分法 FDM 的錯誤響應與泰勒展開證明
為簡化推導，我們觀察一維情況下（ $$y$$ 方向同理）的 FDM 特徵值。根據 Step 1 的泰勒展開，我們知道：

$$\lambda^{fdm} \approx -k_x^2 + \frac{\Delta x^2}{12} k_x^4$$

將其代入變形係數的公式中：

$$\hat{u}^{fdm} = \frac{1}{\lambda^{fdm}} = \frac{1}{-k_x^2 + \frac{\Delta x^2}{12} k_x^4}$$

我們把精確特徵值 $$-k_x^2$$ 提出來：

$$\hat{u}^{fdm} = \frac{1}{-k_x^2 \left( 1 - \frac{\Delta x^2}{12} k_x^2 \right)} = \left( \frac{1}{-k_x^2} \right) \cdot \left( \frac{1}{1 - \frac{\Delta x^2}{12} k_x^2} \right)$$

注意到前項 $$\frac{1}{-k_x^2}$$ 正是精確的 $$\hat{u}^{spec}$$ 。對於後項，利用微積分中的幾何級數近似展開 $$\frac{1}{1-z} \approx 1+z$$（當 $$z$$ 甚小時）：

$$\hat{u}^{fdm} \approx \hat{u}^{spec} \cdot \left( 1 + \frac{\Delta x^2}{12} k_x^2 \right)$$

* 誤差的物理與數學意義
因為 $$\Delta x^2 > 0$$ 且波數平方 $$k_x^2 > 0$$ ，所以括號內的乘子 $$\left( 1 + \frac{\Delta x^2}{12} k_x^2 \right)$$ 必然大於 1。
這在數學上嚴格證明了：

$$\left| \hat{u}^{fdm} \right| > \left| \hat{u}^{spec} \right|$$

* 物理意義：這意味著 FDM 的網格結構在高頻受力下顯得「太軟了」。當遭遇高頻空間擾動（ $$k_x$$ 很大）時，乘子會顯著大於 1，FDM 無法提供足夠的「數值恢復力」，導致計算出的網格位移量被異常放大，超過了真實物理的合理範圍。

##### $$\color{red}{誤差去耦的終極實現與 AI 訓練價值}$$

綜合上述推導，我們透過逆離散正弦轉換 (IDST) 將頻譜係數轉回實數空間，得到兩種解：
高精度：

$$u_{spec} = \mathcal{IDST} \left( \frac{\hat{f}_{m,n}}{\lambda_{m,n}^{spec}} \right)$$

FDM 粗糙解： 

$$u_{fdm} = \mathcal{IDST} \left( \frac{\hat{f}_{m,n}}{\lambda_{m,n}^{fdm}} \right)$$

兩者相減所得的空間殘差（Residual）：

$$\Delta u(x,y) = |u_{spec} - u_{fdm}|$$

##### $$\color{blue}{step2 的結論}$$

這張 $$\Delta u$$ 誤差圖，內部絕對沒有夾雜任何時間步進（如 Euler 或 Runge-Kutta 方法）所產生的混合誤差。它透過反矩陣操作，完美且純粹地捕捉了「FDM 網格過度響應高頻特徵」的空間扭曲現象。


link:https://colab.research.google.com/drive/15TOTS_xZIY6lpomE8A8ijzpu-E3o-zYn?usp=sharing

<br></br>
<br></br>
<br></br>


#### Step 3: Heat Equation 熱傳導方程 —— 動態極限壓力測試 (Dynamic Pure Diffusion)
<br></br>
作用：在完成 Poisson 方程的靜態空間去耦後，我們重新引入時間變數 $t$。本階段的核心戰略是：透過頻域的精確時間解析積分，人為地將「數值時間步進（如 Euler 或 Runge-Kutta 方法）的截斷誤差」強制歸零。如此一來，在任何時間 $$T$$ 觀測到的誤差，皆 $$100\%$$ 源自於空間特徵值 $$\lambda$$ 的差異，讓我們能精確追蹤 FDM 空間誤差在動態衰減過程中的演化軌跡。

##### $$\color{red}{數學定義與利用正交性之 ODE 解耦 (Orthogonal Decoupling)}$$

考慮二維純擴散動態方程，伴隨 Dirichlet 零邊界條件：

$$\frac{\partial u}{\partial t} = D \nabla^2 u$$

我們將未知函數 $$u(x,y,t)$$ 展開為二維傅立葉正弦級數（對應離散正弦轉換 DST），令 $$k_m = \frac{m\pi}{L}, k_n = \frac{n\pi}{L}$$ ：

$$u(x,y,t) = \sum_{m=1}^{N}\sum_{n=1}^{N} \hat{u}_{m,n}(t) \sin(k_m x) \sin(k_n y)$$

將此級數代入偏微分方程：

$$\sum_{m,n} \frac{d \hat{u}_{m,n}(t)}{dt} \sin(k_m x) \sin(k_n y) = D \sum_{m,n} \hat{u}_{m,n}(t) \left[ \nabla^2 \sin(k_m x) \sin(k_n y) \right]$$

由於空間算子 $$\nabla^2$$ 作用於正弦基底會產生特徵值 $$\lambda_{m,n}$$，右式化為：

$$D \sum_{m,n} \hat{u}_{m,n}(t) \lambda_{m,n} \sin(k_m x) \sin(k_n y)$$

為了萃取單一模態 $$(p, q)$$ 的方程，我們在等式兩側同乘上測試函數 $$\sin(k_p x) \sin(k_q y)$$，並對整個空間域 $$\Omega = [0,L]\times[0,L]$$ 進行雙重積分。
根據三角函數的正交性（Orthogonality condition）：

$$\int_0^L \sin(k_m x) \sin(k_p x) dx = \frac{L}{2} \delta_{mp}$$

其中 $$\delta_{mp}$$ 為克羅內克δ函數（Kronecker delta）。當且僅當 $$m=p$$ 且 $$n=q$$ 時，積分值不為零。
所有交叉項相互消滅，積分後得到：

$$\left(\frac{L^2}{4}\right) \frac{d \hat{u}_{p,q}(t)}{dt} = D \lambda_{p,q} \hat{u}_{p,q}(t) \left(\frac{L^2}{4}\right)$$

消去常數項，原偏微分方程完美降維成 $N \times N$ 個獨立的一階線性常微分方程 (ODEs)：

$$\frac{d \hat{u}_{m,n}(t)}{dt} = D \lambda_{m,n} \hat{u}_{m,n}(t)$$

##### $$\color{red}{時間積分之精確解析解 (Exact Analytical Time Integration)}$$

對於上述形式的一階 ODE $$\frac{dy}{dt} = ky$$ ，我們可透過分離變數法進行嚴格求解：

$$\frac{d \hat{u}_{m,n}}{\hat{u}_{m,n}} = D \lambda_{m,n} dt$$

兩側同時積分：

$$\int \frac{1}{\hat{u}_{m,n}} d\hat{u}_{m,n} = \int D \lambda_{m,n} dt$$

$$\ln|\hat{u}_{m,n}(t)| = D \lambda_{m,n} t + C$$

取指數並代入初始條件 $t=0$，得出時間演化的精確解析解：

$$\hat{u}_{m,n}(T) = \hat{u}_{m,n}(0) \cdot e^{D \lambda_{m,n} T}$$

其中 $$\hat{u}_{m,n}(0) = \mathcal{DST}\{u(x,y,0)\}$$。

因為我們使用的是真正的指數函數 $e^{D\lambda T}$ 而非 $\Delta t$ 的截斷逼近，此時間推進過程的誤差為絕對的零。最終實數空間解為 $$u(x,y,T) = \mathcal{IDST} \{ \hat{u}_{m,n}(T) \}$$ 。

##### $$\color{red}{數學範例與證明：高頻衰減率的數值遲滯現象 (Mathematical Proof of Decay Retardation)}$$

為了精確證明 FDM 在動態擴散中的幾何失真，我們觀察單一高頻初始條件 $$u(x,y,0) = \sin(k_x x)\sin(k_y y)$$ 。

* A. FDM 特徵值之泰勒展開極限
為簡化推導，觀察一維情況下 FDM 的特徵值：

$$\lambda^{fdm} = \frac{2}{\Delta x^2}(\cos(k_x \Delta x) - 1)$$

引入餘弦函數的麥克勞林展開式（Maclaurin series） $$\cos(\theta) = 1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \frac{\theta^6}{6!} + \dots$$ ：

$$\lambda^{fdm} = \frac{2}{\Delta x^2}\left[ \left( 1 - \frac{k_x^2 \Delta x^2}{2} + \frac{k_x^4 \Delta x^4}{24} - \dots \right) - 1 \right]$$

$$\lambda^{fdm} = -k_x^2 + \frac{\Delta x^2}{12} k_x^4 - \mathcal{O}(\Delta x^4)$$

注意到第一項 $$-k_x^2$$ 即為真實物理特徵值 $$\lambda^{spec}$$。

* B. 誤差放大率之數學證明
將展開後的 $$\lambda^{fdm}$$ 代入時間演化公式，並利用指數相乘律 $$e^{A+B} = e^A e^B$$ ：

$$u_{fdm}(t) = \sin(k_x x) \cdot \exp\left[D\left(\lambda^{spec} + \frac{\Delta x^2}{12} k_x^4\right)t\right]$$

$$u_{fdm}(t) = \underbrace{\left[ \sin(k_x x) e^{D \lambda^{spec} t} \right]}_{u_{spec}(t) \text{ (真實解)}} \cdot \underbrace{\exp\left(D \frac{\Delta x^2}{12} k_x^4 t\right)}_{\text{數值遲滯乘子 (Retardation Multiplier)}}$$

* C. 漸近行為分析 (Asymptotic Behavior)
考察兩者在時間演化下的誤差比例：

$$\frac{u_{fdm}(t)}{u_{spec}(t)} = \exp\left(D \frac{\Delta x^2}{12} k_x^4 t\right)$$

由於 $$D, \Delta x, k_x, t$$ 皆為正數，該指數必然恆大於 $$1$$ 。
更嚴重的是，當時間 $$t \to \infty$$ 或波數 $$k_x$$ 極大時，此誤差乘子會呈指數級數爆炸。
結論：在數學上嚴格證明了，FDM 的算子在高頻處因為「曲率估計不足」，導致散熱與擴散的計算速度過慢。這使得高頻噪聲（數值畸變）在系統中殘留的時間遠長於真實物理情況，此現象即為「數值耗散不足 (Numerical Under-dissipation)」。

##### $$\color{red}{雙重極限壓力測試的頻譜解析與 AI 戰略意義}$$

結合上述理論，我們在 Heat Equation 中設計了兩組極限測試，以完善 AI 的特徵學習：

1. 高頻壓力測試 (Case 1: Mixed Sine)：

   $$u_0 = \sin\left(\frac{\pi x}{L}\right)\sin\left(\frac{\pi y}{L}\right)\left[2 + \sin\left(\frac{4\pi x}{L}\right)\right]$$

   透過積化和差公式，此初始條件可分解為多個獨立頻率的線性組合，其中包含高頻項 $$k_x = \frac{3\pi}{L}$$ 與 $$\frac{5\pi}{L}$$ 。根據 3.3 節的證明，這些高頻模態在 FDM 演化中會產生極其顯著的「衰減遲滯」，導致波形結構的嚴重畸變。此測試為 AI 提供了修復「動態高頻噪聲」的標準標籤。

3. 邊界極限測試 (Case 2: Polynomial $$x(L-x)y(L-y)$$ )：
   若計算一維多項式 $$x(L-x)$$ 的傅立葉正弦轉換係數：

   $$\hat{u}_m(0) = \frac{2}{L}\int_0^L x(L-x)\sin\left(\frac{m\pi x}{L}\right)dx = \frac{4L^2}{m^3 \pi^3} \quad (\text{for odd } m)$$

   其頻譜係數以 $\mathcal{O}(\frac{1}{m^3})$ 的速率衰減。這意味著多項式無法被有限個正弦波完全表示，它是一個無窮級數。
   此測試暴露出 Spectral Method (DST) 的死穴：在離散網格 $$N$$ 下強制截斷無窮級數，會產生微小的吉布斯現象 (Gibbs Phenomenon) 殘差。此案例確保 AI 模型不會對高頻特徵過度擬合 (Overfitting)，並學會處理非三角函數基底的微小修正。


link:https://colab.research.google.com/drive/15TOTS_xZIY6lpomE8A8ijzpu-E3o-zYn?usp=sharing

<br></br>
<br></br>
<br></br>

#### Step 4: Fisher-KPP 方程 —— 最終動態非線性系統 (The Final Boss)

作用：Fisher-KPP 方程是本研究最終的應用目標，廣泛用於描述細胞擴散、基因傳播與腫瘤生長等反應擴散系統。在這個系統中，空間的「線性擴散算子」與局部的「非線性反應算子」強烈耦合。本階段的推導將利用泛函分析與李代數（Lie Algebra）嚴格證明：為何時間步進法（Splitting Method）無可避免地會在波前處製造出「時間與空間複合的幽靈誤差」，從而確立本研究「空間誤差前置去耦戰略」的絕對必要性。

##### $$\color{red}{數學定義與算子競爭 (Operator Competition)}$$

給定 Fisher-KPP 偏微分方程：

$$\frac{\partial u}{\partial t} = \underbrace{D \nabla^2 u}_{\text{算子 } \mathcal{A} \text{ (擴散)}} + \underbrace{\rho u \left(1 - \frac{u}{K}\right)}_{\text{算子 } \mathcal{B} \text{ (反應)}}$$

為簡化後續的數學表示，我們將環境承載力 (Carrying Capacity) $K$ 正規化為 $1$。系統的總體演化可抽象表示為：

$$\frac{\partial u}{\partial t} = (\mathcal{A} + \mathcal{B})u$$

##### $$\color{red}{反應算子的精確解析解 (Exact Logistic Evolution)}$$

若我們暫時關閉擴散算子 ( $$\mathcal{A}=0$$ )，系統退化為純粹的 Logistic 生長常微分方程。我們透過分離變數法進行嚴格積分：

$$\frac{du}{dt} = \rho u(1-u)$$


$$\int \frac{1}{u(1-u)} du = \int \rho \, dt$$

利用部分分式展開 (Partial Fraction Decomposition) 將左式拆解：

$$\int \left( \frac{1}{u} + \frac{1}{1-u} \right) du = \rho t + C$$

$$\ln|u| - \ln|1-u| = \rho t + C \implies \ln\left|\frac{u}{1-u}\right| = \rho t + C$$

取指數，並假設初始狀態 $$u(0) = u_0$$，可定出常數 $$A = e^C = \frac{u_0}{1-u_0}$$ ：

$$\frac{u(t)}{1-u(t)} = \frac{u_0}{1-u_0} e^{\rho t}$$

整理並解出 $$u(t)$$ ：

$$u(t)(1-u_0) = u_0 e^{\rho t} - u(t) u_0 e^{\rho t}$$

$$u(t) \left[ 1 - u_0 + u_0 e^{\rho t} \right] = u_0 e^{\rho t}$$

最終得到精確的解析演化算子 $$\Phi_{\mathcal{B}}^{\Delta t}$$ ：

$$u(t+\Delta t) = \Phi_{\mathcal{B}}^{\Delta t}[u(t)] = \frac{u(t) e^{\rho \Delta t}}{1 + u(t)(e^{\rho \Delta t} - 1)}$$

這意味著，若僅存在反應項，我們能達成「零時間截斷誤差」的完美演化。


##### $$\color{red}{算子分裂法與 BCH 公式之泰勒展開證明 (Splitting\quad and \quad BCH \quad Formula)}$$

既然擴散 $$\mathcal{A}$$（經由 DST 頻域解析）和反應 $$\mathcal{B}$$ （經由 Logistic 公式）皆具備精確演化路徑，真實系統的演化為 $$u(t+\Delta t) = e^{(\mathcal{A}+\mathcal{B})\Delta t} u(t)$$ 。
若我們強行將兩者拆分依序計算（即 Lie-Trotter Splitting $$e^{\mathcal{A}\Delta t} e^{\mathcal{B}\Delta t}$$ ），其誤差來源可透過泰勒級數嚴格展開：

真實演化的泰勒展開：

$$e^{(\mathcal{A}+\mathcal{B})\Delta t} = \mathcal{I} + (\mathcal{A}+\mathcal{B})\Delta t + \frac{1}{2}(\mathcal{A}+\mathcal{B})^2 \Delta t^2 + \mathcal{O}(\Delta t^3)$$

展開平方項 $$(\mathcal{A}+\mathcal{B})^2 = \mathcal{A}^2 + \mathcal{A}\mathcal{B} + \mathcal{B}\mathcal{A} + \mathcal{B}^2$$ 。

分裂演化的泰勒展開：

$$e^{\mathcal{A}\Delta t} e^{\mathcal{B}\Delta t} = \left( \mathcal{I} + \mathcal{A}\Delta t + \frac{1}{2}\mathcal{A}^2 \Delta t^2 \right) \left( \mathcal{I} + \mathcal{B}\Delta t + \frac{1}{2}\mathcal{B}^2 \Delta t^2 \right) + \mathcal{O}(\Delta t^3)$$

$$= \mathcal{I} + (\mathcal{A}+\mathcal{B})\Delta t + \left( \frac{1}{2}\mathcal{A}^2 + \mathcal{A}\mathcal{B} + \frac{1}{2}\mathcal{B}^2 \right) \Delta t^2 + \mathcal{O}(\Delta t^3)$$

將「真實演化」減去「分裂演化」，可得局部截斷誤差 (Local Truncation Error, LTE)：

$$\text{LTE} = \frac{1}{2} (\mathcal{B}\mathcal{A} - \mathcal{A}\mathcal{B}) \Delta t^2 = -\frac{1}{2} [\mathcal{A}, \mathcal{B}] \Delta t^2$$

此即為貝克-坎貝爾-豪斯多夫公式 (BCH Formula) 的一階截斷結果。
數學鐵律：只有當算子對易（ $$[\mathcal{A},\mathcal{B}] = \mathcal{A}\mathcal{B} - \mathcal{B}\mathcal{A} = 0$$ ）時，拆分才沒有 $$\mathcal{O}(\Delta t^2)$$ 的誤差。

##### $$\color{red}{數學範例與證明：Fisher-KPP 對易子誤差之嚴格推導 (Fréchet Derivative)}$$

在非線性系統中，對易子定義為方向導數（弗雷歇導數, Fréchet Derivative）之差：

$$[\mathcal{A},\mathcal{B}]u = \mathcal{A}(\mathcal{B}(u)) - \mathcal{B}'(u)\mathcal{A}(u)$$

令擴散算子 $$\mathcal{A}(u) = D\nabla^2 u$$ ，非線性反應算子 $$\mathcal{B}(u) = f(u) = \rho u(1-u)$$ 。

* 推導步驟 1：計算反應算子的弗雷歇導數 $$\mathcal{B}'(u)$$
對任意擾動 $$v$$ ，導數定義為：

$$\mathcal{B}'(u)v = \lim_{\epsilon \to 0} \frac{f(u+\epsilon v) - f(u)}{\epsilon} = \frac{d}{du}[\rho u(1-u)] \cdot v = \rho(1-2u)v$$

將 $$v$$ 替換為 $$\mathcal{A}(u) = D\nabla^2 u$$ ：

$$\mathcal{B}'(u)\mathcal{A}(u) = \rho(1-2u) [D\nabla^2 u] = \rho D(1-2u)\nabla^2 u$$

* 推導步驟 2：計算先反應後擴散 $$\mathcal{A}(\mathcal{B}(u))$$
將 $$\nabla^2 = \nabla \cdot \nabla$$ 作用於 $$f(u) = \rho u(1-u)$$ 。根據向量微積分的連鎖律 $$\nabla f(u) = f'(u)\nabla u$$ ，以及散度法則 $$\nabla \cdot (\phi \mathbf{V}) = \nabla \phi \cdot \mathbf{V} + \phi \nabla \cdot \mathbf{V}$$ ：

$$\mathcal{A}(\mathcal{B}(u)) = D \nabla \cdot (\nabla (\rho u - \rho u^2))$$

$$= D\rho \nabla \cdot ((1-2u)\nabla u)$$

$$= D\rho \left[ \nabla(1-2u) \cdot \nabla u + (1-2u)\nabla \cdot (\nabla u) \right]$$

$$= D\rho \left[ (-2\nabla u) \cdot \nabla u + (1-2u)\nabla^2 u \right]$$

$$= -2D\rho |\nabla u|^2 + D\rho(1-2u)\nabla^2 u$$

* 推導步驟 3：計算對易子李括號
將兩者相減：

$$[\mathcal{A},\mathcal{B}]u = \left( -2D\rho |\nabla u|^2 + D\rho(1-2u)\nabla^2 u \right) - \left( D\rho(1-2u)\nabla^2 u \right)$$

$$[\mathcal{A},\mathcal{B}]u = -2D\rho |\nabla u|^2$$

* 結論：推導結果 $$-2D\rho |\nabla u|^2$$ 是一個極度震撼的數學事實！這嚴格證明了時間分裂造成的誤差，恰好正比於波前空間斜率的平方 ( $$|\nabla u|^2$$ )。
* 物理意義：在 Fisher-KPP 行進波中，波前（濃度從 1 掉到 0 的交界處）是最陡峭的，也就是 $$|\nabla u|$$ 最大的地方。這在數學上確鑿地證明了：時間步進誤差與空間幾何特徵高度耦合，並會在陡峭的波前處引發指數級的數值災難。

##### $$\color{red}{二階 Strang Splitting 之具體實踐與對稱性抵消 (Symmetry Cancellation)}$$

為了壓制這個由 $$[\mathcal{A},\mathcal{B}]$$ 帶來的巨大誤差，我們放棄 Lie-Trotter 分裂，改採對稱的 Strang Splitting。此方法透過「半步-一步-半步」的結構進行演化：

$$u(t+\Delta t) \approx S_{\Delta t} [u(t)] = \Phi_{\mathcal{B}}^{\Delta t/2} \circ \Phi_{\mathcal{A}}^{\Delta t} \circ \Phi_{\mathcal{B}}^{\Delta t/2} [u(t)]$$

優勢：因為演化算子 $S_{\Delta t}$ 具備時間對稱性（即 $$S_{-\Delta t} = S_{\Delta t}^{-1}$$ ），其泰勒展開式中的偶數階誤差項（包含 $$\Delta t^2$$ 的對易子項）會完美相互抵消。這使得局部截斷誤差 (LTE) 提升至 $$\mathcal{O}(\Delta t^3)$$，全域截斷誤差 (GTE) 提升至 $$\mathcal{O}(\Delta t^2)$$ 。

演算法執行程序：
1. 半步反應：利用 反應算子的精確解析解 (Exact Logistic Evolution) 的 Logistic 解析解推進 $$\frac{\Delta t}{2}$$ 。
2. 一步擴散：利用 2D DST 轉入頻域，套用 Step 3 的精確擴散公式 $$e^{D \lambda_{m,n} \Delta t}$$ ，再用 IDST 轉回空間。
3. 半步反應：對前一步的結果再次套用 Logistic 解析解推進最後的 $$\frac{\Delta t}{2}$$ 。

##### $$\color{blue}{結論}$$

透過 Fisher-KPP 對易子誤差之嚴格推導 (Fréchet Derivative)的對易子證明我們確信，在 Fisher-KPP 系統中，非線性對易子造成的時間誤差 ( $$|\nabla u|^2$$ ) 與 FDM 的空間截斷誤差 ( $$\frac{\partial^4 u}{\partial x^4}$$ ) 全都集中在最陡峭的波前交疊爆發。

如果直接使用 Fisher-KPP 的演化數據去訓練神經網路，AI 接收到的 Loss 訊號將是高度污染的混合體。模型無法分辨眼前的畸變到底是來自「算子分裂的時間對易子延遲」，還是來自「FDM 網格的空間高頻流失」，最終導致卷積核 (Convolutional Kernels) 無法收斂到正確的物理特徵上。

link:https://colab.research.google.com/drive/15TOTS_xZIY6lpomE8A8ijzpu-E3o-zYn?usp=sharing


---

* Step 2 (Poisson)：利用代數逆矩陣剔除時間變數，強迫 AI 專注建立修復 $$\frac{\partial^4 u}{\partial x^4}$$ 空間扭曲的感知野 (Receptive Field)。
* Step 3 (Heat)：利用 $$[\mathcal{A},\mathcal{A}]=0$$ 的純線性系統，教 AI 理解空間誤差在動態擴散中的衰減遲滯現象。
* Step 4 (Fisher-KPP)：當 AI 帶著純粹的「空間修正權重」來到這一步時，它只需精準地拔除 FDM 網格的幾何離散誤差，而將殘留的純時間對易子誤差留給二階 Strang Splitting 去物理壓制，完美實現了傳統數值算法與現代深度學習的最佳分工協作。
