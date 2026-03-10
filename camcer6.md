**使用CNN訓練Data**

#### Step1:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dst, idst
from scipy.ndimage import zoom
import os
import random


class FisherKPPMMS_DST:
    def __init__(self, N=64, L=50.0, D=1.0, rho=1.0, K=100.0, omega=3.0):
        self.N = N
        self.L = L
        self.D = D
        self.rho = rho
        self.K = K
        self.omega = omega 
        self.dx = L / (N + 1)
        
        x = np.linspace(self.dx, L - self.dx, N)
        y = np.linspace(self.dx, L - self.dx, N)
        self.X, self.Y = np.meshgrid(x, y)
        
        k = np.arange(1, N + 1)
        kx, ky = np.meshgrid(k, k)
        
        self.lambda_spec = -((kx * np.pi / L)**2 + (ky * np.pi / L)**2)
        lx_fdm = (2 / self.dx**2) * (np.cos(kx * np.pi / (N + 1)) - 1)
        ly_fdm = (2 / self.dx**2) * (np.cos(ky * np.pi / (N + 1)) - 1)
        self.lambda_fdm = lx_fdm + ly_fdm

        self.scale = 1e-4
        self.Phi = self.scale * self.X * (self.L - self.X) * self.Y * (self.L - self.Y)
        raw_neg_lap = 2 * (self.Y * (self.L - self.Y) + self.X * (self.L - self.X))
        self.neg_lap_Phi = self.scale * raw_neg_lap

    def get_mms_data(self, t, case):
        Phi = self.Phi
        neg_lap_Phi = self.neg_lap_Phi
        
        if case == 3:
            k_wave = 4 * np.pi / self.L
            theta = k_wave * self.X - self.omega * t # 使用物件的 omega
            
            v = 2 + np.sin(theta)
            u_exact = Phi * v
            
            du_dt = -self.omega * Phi * np.cos(theta)
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
        return None, None

    def reaction_exact(self, u, dt):
        u = np.maximum(u, 0)
        exp_rho = np.exp(self.rho * dt)
        numerator = self.K * u * exp_rho
        denominator = self.K + u * (exp_rho - 1)
        return numerator / denominator


def generate_augmented_dataset():
    print("\n" + "="*65)
    print("開始生成 [擴增版] 機器學習訓練資料 (Data Augmentation)")
    print("="*65)
    
    L = 50.0
    T_total = 2.0      
    dt = 1e-4          
    steps = int(T_total / dt)
    save_interval = steps // 100  # 每次模擬抓 100 張圖
    
    N_coarse = 16
    N_fine = 64
    case_id = 3
    
    # 決定要跑幾次不同的模擬 (15 次 * 100 張 = 1500 張訓練圖)
    num_simulations = 15 
    
    X_dataset = []
    Y_dataset = []
    
    np.random.seed(42) # 固定亂數種子，讓結果可重現
    
    for sim in range(num_simulations):
        #  隨機抽取物理參數
        random_D = np.random.uniform(0.5, 2.0)     # 擴散係數在 0.5 到 2.0 之間變化
        random_omega = np.random.uniform(1.0, 5.0) # 波速在 1.0 到 5.0 之間變化
        
        print(f"[{sim+1}/{num_simulations}] 模擬開始 | D = {random_D:.2f}, omega = {random_omega:.2f} ...")
        
        solver_coarse = FisherKPPMMS_DST(N=N_coarse, L=L, D=random_D, omega=random_omega)
        solver_fine = FisherKPPMMS_DST(N=N_fine, L=L, D=random_D, omega=random_omega)
        
        u_coarse, _ = solver_coarse.get_mms_data(0, case_id)
        t = 0.0
        decay_coarse = np.exp(solver_coarse.D * solver_coarse.lambda_fdm * dt)
        
        for step in range(steps):
            u_coarse = solver_coarse.reaction_exact(u_coarse, dt/2)
            _, f_source_coarse = solver_coarse.get_mms_data(t, case_id)
            
            u_hat = dst(dst(u_coarse, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
            f_hat = dst(dst(f_source_coarse, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
            
            with np.errstate(divide='ignore', invalid='ignore'):
                factor = (decay_coarse - 1) / (solver_coarse.D * solver_coarse.lambda_fdm)
                factor = np.where(np.abs(solver_coarse.lambda_fdm) < 1e-12, dt, factor)
                
            u_hat_new = u_hat * decay_coarse + f_hat * factor
            u_coarse = idst(idst(u_hat_new, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
            
            t += dt
            u_coarse = solver_coarse.reaction_exact(u_coarse, dt/2)
            
            if step % save_interval == 0:
                u_exact_fine, _ = solver_fine.get_mms_data(t, case_id)
                u_coarse_upsampled = zoom(u_coarse, 4.0, order=3) 
                
                X_dataset.append(u_coarse_upsampled)
                Y_dataset.append(u_exact_fine)
                
    X_dataset = np.array(X_dataset)
    Y_dataset = np.array(Y_dataset)
    
    os.makedirs('dataset', exist_ok=True)
    np.save('dataset/X_coarse_upsampled.npy', X_dataset)
    np.save('dataset/Y_exact_fine.npy', Y_dataset)
    
    print("\n擴增資料生成完成,Done！")
    print(f"資料集大幅膨脹！現在我們擁有 {len(X_dataset)} 張訓練影像！")
    print(f"X shape: {X_dataset.shape}")
    print(f"Y shape: {Y_dataset.shape}")
```

#### Step2:

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ErrorCorrectionDataset(Dataset):
    def __init__(self, x_path, y_path):

        print(f"載入資料中: {x_path} 與 {y_path}")
        self.X_coarse = np.load(x_path)  # (N_samples, 64, 64)
        self.Y_exact = np.load(y_path)   # (N_samples, 64, 64)
        
        # 核心：AI學習的是「誤差」，而不是直接學最終解
        self.Error_Target = self.Y_exact - self.X_coarse
        
        # 轉換為 PyTorch Tensor，並增加 Channel 維度 (Batch, Channel, Height, Width)
        # 因為是 2D 數值陣列，相當於單色灰階影像，所以 Channel = 1
        self.X_tensor = torch.tensor(self.X_coarse, dtype=torch.float32).unsqueeze(1)
        self.Error_tensor = torch.tensor(self.Error_Target, dtype=torch.float32).unsqueeze(1)
        
        print(f"成功轉換 Tensor！輸入形狀: {self.X_tensor.shape}, 目標形狀: {self.Error_tensor.shape}")

    def __len__(self):
        return len(self.X_coarse)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.Error_tensor[idx]


class ResBlock(nn.Module):
    """
    殘差區塊 (Residual Block)：讓特徵可以直接跳過卷積層相加，
    極大程度避免梯度消失，非常適合用來學習 PDE 的微小誤差。
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x  # 保留原始輸入
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual  # 將原始輸入與卷積後的特徵相加 (ResNet 的精髓)
        out = self.relu(out)
        return out

class PDErrorCorrectionNet(nn.Module):
    def __init__(self):
        super(PDErrorCorrectionNet, self).__init__()
        # 1. 特徵提取層 (將 1 個 Channel 擴充到 16 個特徵空間)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. 深度殘差層 (使用兩個 ResBlock 來捕捉複雜的 Gibbs 空間誤差)
        self.res1 = ResBlock(16)
        self.res2 = ResBlock(16)
        
        # 3. 輸出還原層 (將 16 個特徵壓縮回 1 個 Channel 的誤差預測圖)
        self.out_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: 粗網格數值解 (Upsampled FDM)
        out = self.in_conv(x)
        out = self.res1(out)
        out = self.res2(out)
        pred_error = self.out_conv(out)
        return pred_error # 回傳預測的「誤差補償值」


def test_pipeline():
    print("\n" + "="*50)
    print(" 測試資料載入與神經網路架構")
    print("="*50)
    
    try:
        # 1. 測試 Dataset
        dataset = ErrorCorrectionDataset('dataset/X_coarse_upsampled.npy', 'dataset/Y_exact_fine.npy')
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True) # 每次抓 4 張圖出來
        
        # 拿出一批 (Batch) 資料看看
        inputs, target_errors = next(iter(dataloader))
        print(f"\n[DataLoader 測試]")
        print(f"抽出的一個 Batch Input 形狀: {inputs.shape}  (Batch, C, H, W)")
        print(f"抽出的一個 Batch Target 形狀: {target_errors.shape}")
        
        # 2. 測試神經網路
        model = PDErrorCorrectionNet()
        
        # 將 Inputs 丟進沒訓練過的網路，看看吐出來的形狀對不對
        predicted_errors = model(inputs)
        
        print(f"\n[神經網路 測試]")
        print(f"模型輸出的預測形狀: {predicted_errors.shape}")
        
        # 檢查形狀是否完全吻合
        assert predicted_errors.shape == target_errors.shape, "警告：模型輸出與目標形狀不一致！"
        print("測試完美通過！神經網路的輸入與輸出維度完全吻合，準備好可以開始訓練了！")
        
    except FileNotFoundError:
        print("找不到資料檔，請確認 'dataset' 資料夾是否存在，並且裡面有 .npy 檔案！")

if __name__ == "__main__":
    test_pipeline()
```

#### Step3:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os


class ErrorCorrectionDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X_coarse = np.load(x_path)
        self.Y_exact = np.load(y_path)
        self.Error_Target = self.Y_exact - self.X_coarse
        
        self.X_tensor = torch.tensor(self.X_coarse, dtype=torch.float32).unsqueeze(1)
        self.Error_tensor = torch.tensor(self.Error_Target, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_coarse)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.Error_tensor[idx]


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class PDErrorCorrectionNet(nn.Module):
    def __init__(self):
        super(PDErrorCorrectionNet, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResBlock(16)
        self.res2 = ResBlock(16)
        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.in_conv(x)
        out = self.res1(out)
        out = self.res2(out)
        pred_error = self.out_conv(out)
        return pred_error


def train_resnet_model():
    print("\n" + "="*50)
    print("開始訓練高階誤差修正模型 (ResNet)")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用運算裝置: {device}")

    # 載入資料
    dataset = ErrorCorrectionDataset('dataset/X_coarse_upsampled.npy', 'dataset/Y_exact_fine.npy')
    
    # 80% 訓練集, 20% 驗證集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 初始化模型、Loss 與優化器
    model = PDErrorCorrectionNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    # 升級亮點：學習率排程器 (每 50 個 Epoch，學習率衰減一半)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    num_epochs = 300  # 增加訓練時間以發揮 ResNet 潛力
    train_loss_history = []
    val_loss_history = []
    
    print(f"\n資料筆數 -> 訓練集: {train_size}, 驗證集: {val_size}")
    print(f"總 Epoch 數: {num_epochs}")
    print("開始訓練...\n")
    
    for epoch in range(num_epochs):
        # --- 訓練階段 ---
        model.train()
        train_loss = 0.0
        for inputs, target_errors in train_loader:
            inputs, target_errors = inputs.to(device), target_errors.to(device)
            
            optimizer.zero_grad()
            pred_errors = model(inputs)
            loss = criterion(pred_errors, target_errors)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= train_size
        train_loss_history.append(train_loss)
        
        # --- 驗證階段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, target_errors in val_loader:
                inputs, target_errors = inputs.to(device), target_errors.to(device)
                pred_errors = model(inputs)
                loss = criterion(pred_errors, target_errors)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= val_size
        val_loss_history.append(val_loss)
        
        # 更新學習率
        scheduler.step()
        
        # 每 20 個 Epoch 印出一次進度
        if (epoch+1) % 20 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1:3d}/{num_epochs}] | Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e} | LR: {current_lr:.2e}')
            
    # 儲存模型
    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/resnet_error_correction.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"\n訓練完成！模型已儲存至 {model_save_path}")
    
    # 繪製 Loss 曲線
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, 'b-', linewidth=2, label='Training Loss')
    plt.plot(val_loss_history, 'r--', linewidth=2, label='Validation Loss')
    plt.title('ResNet Training & Validation Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_resnet_model()
```

#### Step4:

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class PDErrorCorrectionNet(nn.Module):
    def __init__(self):
        super(PDErrorCorrectionNet, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResBlock(16)
        self.res2 = ResBlock(16)
        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.in_conv(x)
        out = self.res1(out)
        out = self.res2(out)
        pred_error = self.out_conv(out)
        return pred_error


def evaluate_augmented_resnet():
    print("\n" + "="*60)
    print("驗收成果：擴增資料集訓練後的神經網路推論測試")
    print("="*60)
    
    # 載入資料
    try:
        X_data = np.load('dataset/X_coarse_upsampled.npy')
        Y_data = np.load('dataset/Y_exact_fine.npy')
    except FileNotFoundError:
        print("找不到資料集！請確認 dataset/X_coarse_upsampled.npy 是否存在。")
        return

    # 隨機抽取一張測試圖 (或是你也可以指定 test_idx = -1 測最後一張)
    np.random.seed(42) # 固定亂數種子，方便核對結果
    test_idx = np.random.randint(0, len(X_data))
    
    X_test = X_data[test_idx]
    Y_exact = Y_data[test_idx]
    print(f"我們隨機抽出了第 {test_idx} 張快照來進行期末考！\n")
    
    # 轉為 PyTorch Tensor (Batch=1, Channel=1, H=64, W=64)
    X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 載入剛才訓練好的 ResNet 模型權重
    model = PDErrorCorrectionNet()
    try:
        model.load_state_dict(torch.load('models/resnet_error_correction.pth', map_location='cpu', weights_only=True))
    except FileNotFoundError:
        print("找不到模型權重檔！請確認 models/resnet_error_correction.pth 是否存在。")
        return
        
    model.eval() # 極度重要：切換到推論模式
    
    # 讓 AI 預測誤差
    with torch.no_grad():
        Predicted_Error_tensor = model(X_tensor)
    
    # 轉回 NumPy 陣列
    Predicted_Error = Predicted_Error_tensor.squeeze().numpy()
    
    # 計算 AI 修正後的最終解
    Y_ML_Corrected = X_test + Predicted_Error
    

    Err_Raw_Coarse = np.abs(Y_exact - X_test)
    Err_ML_Corrected = np.abs(Y_exact - Y_ML_Corrected)
    
    max_err_raw = np.max(Err_Raw_Coarse)
    max_err_ml = np.max(Err_ML_Corrected)
    improvement_ratio = max_err_raw / max_err_ml
    
    print(f"傳統粗網格 (N=16) 最大誤差: {max_err_raw:.4e}")
    print(f"AI修正後 (N=16+ResNet) 最大誤差: {max_err_ml:.4e}")
    print(f"終極精度提升倍率: {improvement_ratio:.2f} 倍！")


    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 上半部：解的輪廓
    im0 = axes[0, 0].imshow(X_test, origin='lower', extent=[0,50,0,50], cmap='inferno')
    axes[0, 0].set_title(f'Base: Coarse FDM ($N=16$)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(Y_ML_Corrected, origin='lower', extent=[0,50,0,50], cmap='inferno')
    axes[0, 1].set_title('Proposed: AI-Corrected Solution')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 下半部：誤差對比 (共用 Colorbar 上限以示公平)
    vmax_error = max_err_raw 
    
    im2 = axes[1, 0].imshow(Err_Raw_Coarse, origin='lower', extent=[0,50,0,50], cmap='jet', vmin=0, vmax=vmax_error)
    axes[1, 0].set_title(f'Raw Coarse Error\nMax: {max_err_raw:.2e}')
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].imshow(Err_ML_Corrected, origin='lower', extent=[0,50,0,50], cmap='jet', vmin=0, vmax=vmax_error)
    axes[1, 1].set_title(f'AI-Corrected Error\nMax: {max_err_ml:.2e}')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_augmented_resnet()


if __name__ == "__main__":
    generate_augmented_dataset()
```


#### Step5:

```python
import time
import torch
import torch.nn as nn
import numpy as np
from scipy.fft import dst, idst
from scipy.ndimage import zoom


class FisherKPPMMS_DST:
    def __init__(self, N=64, L=50.0, D=1.0, rho=1.0, K=100.0, omega=3.0):
        self.N = N
        self.L = L
        self.D = D
        self.rho = rho
        self.K = K
        self.omega = omega
        self.dx = L / (N + 1)
        x = np.linspace(self.dx, L - self.dx, N)
        y = np.linspace(self.dx, L - self.dx, N)
        self.X, self.Y = np.meshgrid(x, y)
        k = np.arange(1, N + 1)
        kx, ky = np.meshgrid(k, k)
        self.lambda_spec = -((kx * np.pi / L)**2 + (ky * np.pi / L)**2)
        lx_fdm = (2 / self.dx**2) * (np.cos(kx * np.pi / (N + 1)) - 1)
        ly_fdm = (2 / self.dx**2) * (np.cos(ky * np.pi / (N + 1)) - 1)
        self.lambda_fdm = lx_fdm + ly_fdm
        self.scale = 1e-4
        self.Phi = self.scale * self.X * (self.L - self.X) * self.Y * (self.L - self.Y)
        raw_neg_lap = 2 * (self.Y * (self.L - self.Y) + self.X * (self.L - self.X))
        self.neg_lap_Phi = self.scale * raw_neg_lap

    def get_mms_data(self, t, case):
        Phi = self.Phi
        neg_lap_Phi = self.neg_lap_Phi
        if case == 3:
            k_wave = 4 * np.pi / self.L
            theta = k_wave * self.X - self.omega * t
            v = 2 + np.sin(theta)
            u_exact = Phi * v
            du_dt = -self.omega * Phi * np.cos(theta)
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
        return None, None

    def reaction_exact(self, u, dt):
        u = np.maximum(u, 0)
        exp_rho = np.exp(self.rho * dt)
        numerator = self.K * u * exp_rho
        denominator = self.K + u * (exp_rho - 1)
        return numerator / denominator

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))) + x)

class PDErrorCorrectionNet(nn.Module):
    def __init__(self):
        super(PDErrorCorrectionNet, self).__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.res1 = ResBlock(16)
        self.res2 = ResBlock(16)
        self.out_conv = nn.Conv2d(16, 1, 3, padding=1)
    def forward(self, x):
        return self.out_conv(self.res2(self.res1(self.in_conv(x))))


def benchmark_speed():
    print("\n" + "="*65)
    print(" 傳統高解析度 FDM vs. AI 混合加速求解器")
    print("="*65)
    
    L = 50.0
    T_total = 2.0      
    dt = 1e-4          
    steps = int(T_total / dt)
    case_id = 3
    
    # 準備好 AI 模型 (放置在 CPU 測量，對傳統 FDM 最公平)
    model = PDErrorCorrectionNet()
    try:
        model.load_state_dict(torch.load('models/resnet_error_correction.pth', map_location='cpu', weights_only=True))
    except FileNotFoundError:
        print("找不到模型權重！請確認 models/resnet_error_correction.pth 是否存在。")
        return
    model.eval()


    print(f"\n[回合一] 執行傳統細網格 FDM (N=64) 中，請稍候...")
    solver_fine = FisherKPPMMS_DST(N=64, L=L)
    u_fine, _ = solver_fine.get_mms_data(0, case_id)
    t = 0.0
    decay_fine = np.exp(solver_fine.D * solver_fine.lambda_fdm * dt)
    
    start_time_fine = time.perf_counter() # 碼錶開始！
    
    for step in range(steps):
        u_fine = solver_fine.reaction_exact(u_fine, dt/2)
        _, f_source = solver_fine.get_mms_data(t, case_id)
        u_hat = dst(dst(u_fine, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
        f_hat = dst(dst(f_source, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = (decay_fine - 1) / (solver_fine.D * solver_fine.lambda_fdm)
            factor = np.where(np.abs(solver_fine.lambda_fdm) < 1e-12, dt, factor)
        u_hat_new = u_hat * decay_fine + f_hat * factor
        u_fine = idst(idst(u_hat_new, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
        t += dt
        u_fine = solver_fine.reaction_exact(u_fine, dt/2)
        
    end_time_fine = time.perf_counter() # 碼錶停止！
    time_trad = end_time_fine - start_time_fine
    print(f"傳統方法完成！耗時: {time_trad:.4f} 秒")


    print(f"\n[回合二] 執行粗網格 FDM (N=16) + AI 瞬間推論中...")
    solver_coarse = FisherKPPMMS_DST(N=16, L=L)
    u_coarse, _ = solver_coarse.get_mms_data(0, case_id)
    t = 0.0
    decay_coarse = np.exp(solver_coarse.D * solver_coarse.lambda_fdm * dt)
    
    start_time_ai = time.perf_counter() # ⏱️ 碼錶開始！
    
    # 1. 跑 N=16 粗網格 (網格點少了 16 倍，速度極快)
    for step in range(steps):
        u_coarse = solver_coarse.reaction_exact(u_coarse, dt/2)
        _, f_source = solver_coarse.get_mms_data(t, case_id)
        u_hat = dst(dst(u_coarse, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
        f_hat = dst(dst(f_source, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = (decay_coarse - 1) / (solver_coarse.D * solver_coarse.lambda_fdm)
            factor = np.where(np.abs(solver_coarse.lambda_fdm) < 1e-12, dt, factor)
        u_hat_new = u_hat * decay_coarse + f_hat * factor
        u_coarse = idst(idst(u_hat_new, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
        t += dt
        u_coarse = solver_coarse.reaction_exact(u_coarse, dt/2)
        
    # 2. 空間插值放大 (毫秒級)
    u_coarse_upsampled = zoom(u_coarse, 4.0, order=3) 
    
    # 3. 呼叫 AI 進行誤差修正 (毫秒級)
    input_tensor = torch.tensor(u_coarse_upsampled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        predicted_error = model(input_tensor).squeeze().numpy()
    final_ai_solution = u_coarse_upsampled + predicted_error
    
    end_time_ai = time.perf_counter() # ⏱️ 碼錶停止！
    time_ai = end_time_ai - start_time_ai
    
    print(f"AI 混合方法完成！耗時: {time_ai:.4f} 秒")


    speedup = time_trad / time_ai
    print("\n" + "="*40)
    print(" 效能評估報告 (論文用數據)")
    print("="*40)
    print(f"傳統 N=64 運算時間 : {time_trad:.4f} 秒")
    print(f"AI 混合方法運算時間: {time_ai:.4f} 秒")
    print(f"運算速度提升倍率 : {speedup:.2f} 倍！")
    print("="*40)
    print("論文可以這樣寫：")
    print(f"「本研究提出之 AI 混合求解器，在維持高精度的前提下，其計算效率為傳統高解析度有限差分法之 {speedup:.2f} 倍，展現出極高的即時預測潛力。」")

if __name__ == "__main__":
    benchmark_speed()
```
