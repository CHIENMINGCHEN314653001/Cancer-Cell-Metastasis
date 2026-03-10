Method of Manufactured Solutions,人造解方法(先射箭，再畫靶（逆向工程）)

在 Science Machine Learning，訓練神經網路需要大量的 Ground Truth（絕對正確的標準答案） 來當作標籤（Label Y）。
  * 一般的情況： 因為不知道標準答案，只能把 FDM 的網格開到超級大（例如 N=1024），跑上好幾天，把這個看起來比較準的數值解當作標準答案餵給 AI。這不僅耗時，而且答案本身還是帶有微小的數值誤差。
  * 挑戰： 因為使用了 MMS，標準答案是數學公式直接推出來的，只需要把座標 $$(x,y)$$ 代入公式，很快就能產生毫無數值誤差的完美 $$Y_{exact}$$。

**目前狀況**
1. FDM 的天花板： 空間只有二階精度 $$O(\Delta x^2)$$ 。如果要更精確，網格 $$N$$ 就要變得非常大，導致計算速度呈指數級變慢。
2. Spectral (DST) 的天花板： 遇到邊界不吻合（像是多項式這類的）時，會產生致命的 Gibbs Phenomenon，導致精度崩潰

   共同問題:時間步長 ($$dt$$) 的天花板：Operator Splitting（算子分裂法）有時間截斷誤差，這使得你的總誤差卡在 $$10^{-7}$$ ，逼得你必須用極小的 $$dt=10^{-5}$$ 來跑，這非常耗時。

Target:改善傳統的FDM，用CNN訓練!!  AI預測 (Forward Pass)--->預測 (Forward Pass)--->修正調整 (Backpropagation & Gradient Descent)

*步驟:
  * 生成資料
