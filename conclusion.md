 #### 研究核心動機 (Core Objective)

研究旨在解決數值計算中「計算效率」與「精確度」的權衡問題。
1. 有限差分法 (FDM)：計算成本低、易於並行化且能處理複雜幾何，但在粗網格下會產生顯著的空間截斷誤差 $$\mathcal{O}(\Delta x^2)$$。
2.  頻譜法 (Spectral Method) ：在處理平滑函數時具備「指數級收斂」特性，但在非週期性邊界或非線性強耦合下計算限制較多。

為了系統性地定位誤差源，本研究採取由繁入簡、再由簡入繁的「去耦 (Decoupling)」戰略。

 #### 第一階段：Fisher-KPP 方程 —— 動態非線性系統分析
* 物理意義：描述細胞擴散與群體生長的反應擴散過程。
* 數學定義： $$\frac{\partial u}{\partial t} = D \nabla^2 u + \rho u (1 - \frac{u}{K})$$
* 數值架構：採用  二階 Strang Splitting (史特朗分裂法) 。
    * 將算子拆分為： $$\exp(R \frac{\Delta t}{2}) \exp(D \Delta t) \exp(R \frac{\Delta t}{2})$$
* 關鍵發現：
    * 證實了總誤差受限於空間算子：當波數 (Wavenumber) 增加時，FDM 的空間誤差會以 $$\frac{\partial^4 u}{\partial x^4}$$ 的速率噴發。
    *  戰略價值 ：確定了研究目標應優先處理「空間截斷誤差」而非時間步進誤差。

 #### 第二階段：Poisson 方程 —— 空間誤差完全去耦
* 物理意義：靜態空間平衡系統。
* 數學定義： $$\nabla^2 u = f(x, y)$$
* 研究亮點：
    *  時間暫停 ：移除 $$\frac{\partial u}{\partial t}$$，使誤差 $$100\%$$ 來源於空間離散化。
    *  特徵值診斷 ：利用離散正弦轉換 (DST) 將問題轉入頻域，對比兩者特徵值：
        * Spectral: $$\lambda_{m,n} = - [(\frac{m\pi}{L})^2 + (\frac{n\pi}{L})^2]$$
        * FDM: $$\lambda_{m,n} = \frac{2}{\Delta x^2}(\cos\frac{m\pi\Delta x}{L} - 1) + \frac{2}{\Delta y^2}(\cos\frac{n\pi\Delta y}{L} - 1)$$
    *  戰略價值 ：成功產出了「最乾淨」的訓練數據，讓 AI 只專注學習 $$\lambda_{fdm} \to \lambda_{spec}$$ 的映射關係。

 #### 第三階段：Heat Equation 熱傳導方程 —— 極限壓力測試
*  物理意義 ：純擴散動態過程。
*  數學定義 ： $$\frac{\partial u}{\partial t} = D \nabla^2 u$$
*  雙重極限測試 ：
    1.   高頻測試 (Case 1: Mixed Sine) ：測試 FDM 對劇烈震盪訊號的處理能力，揭示了 FDM 在解析高頻細節時的數值畸變。
    2.   多項式測試 (Case 2: $$x(L-x)y(L-y)$$ ) ：測試 Spectral 對非三角函數基底的逼近極限，揭示了級數截斷產生的微小殘差。
*  戰略價值 ：建立 AI 的「廣泛泛化能力」，確保模型不只能修復 $$\sin$$ 波，也能處理一般性的多項式分佈。

利用了分離變數法 (Separation of Variables) 與 傅立葉正弦轉換 (DST)，把原本複雜的偏微分方程 (PDE) 變成了一堆極度簡單的常微分方程 (ODE)。

>#### 熱傳導方程式的頻域精確解推導 (Exact Integration in Frequency Domain)
>
>#### 1. 問題定義 (Problem Statement)
>我們考慮二維無源項的熱傳導方程式 (Heat Equation)：
> $$\frac{\partial u}{\partial t} = D \nabla^2 u = D \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$
>* 邊界條件 (Dirichlet BCs)：在邊界上 $$u(x,y,t) = 0$$。
>* 初始條件 (Initial Condition)： $$u(x,y,0) = u_0(x,y)$$。
>
>#### 2. 空間基底展開 (Spatial Expansion via DST)
>因為滿足零邊界條件，我們可以將解 $$u(x,y,t)$$ 展開為二維傅立葉正弦級數 (對應程式碼中的 `dst`)：
>
> $$u(x, y, t) = \sum_{m=1}^{N} \sum_{n=1}^{N} \hat{u}_{m,n}(t) \sin\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi y}{L}\right)$$
>
> $$其中, \hat{u}_{m,n}(t)是隨時間變化的頻域係數。$$
>
>在 $$t=0$$ 時，初始係數為（對應 `u0_hat = dst(...)`）：
>
>$$\hat{u}_{m,n}(0) = \mathcal{DST}\{u_0(x,y)\}$$
>
>#### 3. 空間微分算子的特徵值轉換 (Eigenvalue Mapping)
>把 Laplacian 算子 $\nabla^2$ 作用在正弦基底上時，它會變成一個簡單的標量乘法（特徵值 $$\lambda$$）。這裡有兩種不同的空間離散化策略：
>
>A. 頻譜法 (Spectral Method - 精確空間微分)
>連續空間的精確二次微分：
>
> $$\nabla^2 \left( \sin\frac{m\pi x}{L} \sin\frac{n\pi y}{L} \right) = -\left[ \left(\frac{m\pi}{L}\right)^2 + \left(\frac{n\pi}{L}\right)^2 \right] \sin\frac{m\pi x}{L} \sin\frac{n\pi y}{L}$$
>
>因此，頻譜特徵值為（對應 `self.lambda_spec`）：
>
> $$\lambda_{m,n}^{spec} = -\left[ \left(\frac{m\pi}{L}\right)^2 + \left(\frac{n\pi}{L}\right)^2 \right]$$
>
>B. 有限差分法 (FDM - 近似空間微分)
>若使用二階中心差分法，在離散網格點上的特徵值為（對應 `self.lambda_fdm`）：
>
> $$\lambda_{m,n}^{fdm} = \frac{2}{\Delta x^2}\left(\cos\frac{m\pi\Delta x}{L} - 1\right) + \frac{2}{\Delta y^2}\left(\cos\frac{n\pi\Delta y}{L} - 1\right)$$
>
>#### 4. 轉換為常微分方程 (Reduction to ODEs)
>將展開式代回原本的熱傳導方程式，並利用正弦函數的正交性，我們可以將 PDE 解耦成 $N \times N$ 個獨立的一階線性常微分方程 (ODE)：
>
> $$\frac{d}{dt} \hat{u}_{m,n}(t) = D \cdot \lambda_{m,n} \cdot \hat{u}_{m,n}(t)$$
>
>#### 5. 時間積分的精確解 (Exact Time Integration)
>對於上述的一階 ODE $$\frac{dy}{dt} = ky$$，其解析解為 $y(t) = y(0)e^{kt}$$ 。
>
>因此，可以直接算出任意時間 $T$ 的頻域係數，完全不需要切分 $\Delta t$ 迴圈：
>
> $$\hat{u}_{m,n}(T) = \hat{u}_{m,n}(0) \cdot e^{D \lambda_{m,n} T}$$
>
>這行數學式完美對應了你程式碼中最核心的那一行：
>`uT_hat = u0_hat * np.exp(self.D * lambdas * T)`
>
>#### 6. 轉回空間域 (Inverse Transform)
>最後，將 $$T$$ 時刻的頻域係數 $$\hat{u}_{m,n}(T)$$ ， 透過逆離散正弦轉換 (IDST) 組合回空間域：
>
> $$u(x,y,T) = \mathcal{IDST} \{ \hat{u}_{m,n}(T) \}$$
>（對應程式碼中的 `uT = idst(...)`）。
>
>1. 無時間誤：
>   「因為方程式中沒有非線性的反應項 (Reaction term)，所以我在頻域中解 $$\frac{d\hat{u}}{dt} = D\lambda\hat{u}$$ 時，使用的是指數函數的解析解 ($e^{D\lambda T}$)，這代表時間推進的誤差為絕對的零。」
>
>2. 兩種方法的唯一差異：
>   「既然時間誤差為零，那麼 FDM 與 Spectral 解出來的結果為什麼會不一樣？原因完全且唯一地來自於 $$\lambda_{m,n}^{fdm}$$ 與 $$\lambda_{m,n}^{spec}$$ 的差異。FDM 在高頻（ $$m, n$$ 很大）時，其 $$\cos$$ 逼近公式會與真實的二次方>曲線產生極大的偏離，導致高頻波的衰減速度算錯。」
>

<br></br>

 #### 數學嚴謹度證明 (Verification & Validation)

本研究所有模擬數據均經過「製造解方法 (MMS)」驗證，確保數據集的準確性。

 #### 1. 時間誤差收斂證明
透過泰勒展開式證明  Strang Splitting  的二階特性：
*  局部截斷誤差 (LTE) ： $$\mathcal{O}(\Delta t^3)$$。
*  全域截斷誤差 (GTE) ： $$\mathcal{O}(\Delta t^2)$$。
*  驗證結果 ：在雙時間收斂測試中，誤差與 $$\Delta t$$ 呈現明確的二次方關係（倍率為 4），證實了時間步進邏輯的正確性。

 #### 2. 空間誤差收斂證明
針對 Poisson 與 Heat Equation 進行多網格測試 ($$N=16$$ 至 $$512$$)：
*  FDM 表現 ：在 Log-Log 圖中，誤差斜率 (Slope) 精確落在  2.0 ，符合二階中心差分理論。
*  Spectral 表現 ：始終維持在  機器精度 ($$10^{-14} \sim 10^{-16}$$) 。
*  結論 ：數據產生器具備極高的可靠度，產出的差值 (Residual) 是純粹的數值誤差。


<br></br>

#### 小抄
關於 $$Heat Equation$$ 跟 $$poisson equation$$ 差

>在偏微分方程 (PDE) 誤差修正研究中， Poisson 方程和 Heat Equation (熱傳導方程) 就像是為 AI 準備的兩個不同科目的期中考。
>
>
>
>#### 1\. 數學本質與物理直覺的差異
>
>| 特性 | Poisson 方程 (泊松方程) | Heat Equation (熱傳導方程) |
>| :--- | :--- | :--- |
>| 數學公式 | $\nabla^2 u = f(x, y)$ | $\frac{\partial u}{\partial t} = D \nabla^2 u$ |
>| 時間變數 ($t$) | 沒有。 它是純靜態的。 | 有。 它是隨時間演化的動態過程。 |
>| 物理直覺 | 「受力平衡的狀態」<br>想像一張彈性薄膜，你用手指 ( $f$ ) 戳它，薄膜變形後達到靜止不動的最終形狀 ( $u$ )。 | 「逐漸撫平的過程」<br> 想像一塊凹凸不平的冰塊 ( $u_0$ ) 放在室溫下，慢慢融化變平的過程。 |
>| 核心動作 | 尋找源頭 $f$ 所造成的最終空間分佈。 | 觀察初始狀態 $u_0$ 如何隨時間擴散、衰減。 |
>
>
>#### 2\. 在你的AI 誤差修正研究中，它們分別扮演什麼角色？
>
>
>
> 角色一：Poisson 方程 —— 「空間扭曲的靜態照相機」
>
>  * 你在做什麼：給定一個亂七八糟的源頭 $f$，看 FDM 算出來的 $u_{fdm}$ 和完美的 $u_{spec}$ 差多少。
>  * 它的任務：「絕對的空間誤差去耦」。
>  * 對 AI 的意義：因為完全沒有時間 $t$，AI 可以 100% 專心地看著畫面說：「喔！原來 FDM 只要遇到這種弧度（二次微分），就會產生這種形狀的誤差！」這就像是讓 AI 學習認字，是最基礎、最乾淨的特徵訓練。
>
> 角色二：Heat Equation —— 「極限特徵的動態測試台」
>
>  * 你在做什麼：給定一個奇怪的初始圖案 $u_0$（高頻波或多項式），直接跳到時間 $T=5.0$，看這張圖「融化」到一半時，雙方算出來的結果差多少。
>  * 它的任務：尋找演算法的死穴。
>  * 對 AI 的意義：雖然 Heat Equation 有時間 $t$，但因為我們用了超強的「頻域精確積分 ($e^{D\lambda T}$)」，我們人為地把時間誤差變成了 0。這時候：
>      * Case 1 (高頻波) 告訴 AI：注意！FDM 在波浪很密集的地方會大翻車，你要學會把這些高頻的鋸齒修平。
>      * Case 2 (多項式) 告訴 AI：注意！Spectral 雖然神，但在處理特定邊界時也會有截斷瑕疵（微小波紋）。
>      * 這是進階題，確保你的 AI 未來在面對真實且複雜的 Fisher-KPP 模擬時，不會因為沒看過高頻波而崩潰。
>
>### 3\. 終極對比：為什麼不直接用 Fisher-KPP 練 AI？
>
>我們把你的終極目標 Fisher-KPP 方程 加進來比較，你就會豁然開朗：
>
>$$\frac{\partial u}{\partial t} = \underbrace{D \nabla^2 u}_{\text{擴散 (Heat)}} + \underbrace{\rho u \left(1 - \frac{u}{K}\right)}_{\text{非線性反應}}$$
>
>  * Fisher-KPP 是「大魔王」：有空間擴散、有非線性反應、還必須切 $\Delta t$ 一步步算，時間誤差和空間誤差全部攪和在一起（Strang Splitting 產生的 $\mathcal{O}(\Delta t^2)$ 與 $\mathcal{O}(\Delta x^2)$）。如果你直接拿這個給 AI 學，AI 會「走火入魔」，因為它分不清誤差是空間造成的還是時間造成的。
>  * Poisson 是「把大魔王定格」：抽掉時間與反應，只留下空間算子 $\nabla^2$，讓 AI 專心學「空間修正」。
>  * Heat 是「大魔王的純擴散分身」：抽掉反應，利用頻率公式消滅時間誤差，專門測試高頻與邊界極限。

