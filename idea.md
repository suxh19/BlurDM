在 $BlurDM$ 中，模糊是由于曝光时间 $\alpha$ 的累积 。在 CT 中，我们可以将全采样（Full-view）视为在一个完整的角度区间 $[0, \Phi]$ 内的连续积分，而有限角（Limited-angle）则是由于扫描角度范围的减少导致的信息缺失。



以下是模仿 $BlurDM$ 公式 (1)-(3) 的**角度退化扩散模型 (Angle-Degradation Diffusion Model)** 的数学推导：

------

### 1. 物理背景与全采样定义 (Analogy to $I_0$)

在 $BlurDM$ 中，$I_0$ 代表极短曝光下的清晰图像 。在我们的模型中，定义 $S_0$ 为全角度（如 180°）下的理想投影正弦图（Sinogram）：



$$S_0 = \int_{\theta=0}^{\Phi_{max}} P(\theta) d\theta$$

其中 $P(\theta)$ 是在角度 $\theta$ 处的瞬时投影。此时图像重建质量最高，没有任何有限角伪影。

### 2. 角度退化过程 (Forward Angle Diffusion)

我们需要建立一个从全角度 $\Phi_{max}$ 逐步减少到有限角度 $\Phi_t$ 的过程。定义 $\phi_t$ 为在扩散步长 $t$ 时的有效采样角度范围，且满足 $\Phi_{max} = \phi_0 > \phi_1 > \dots > \phi_T$。

#### 步长 $t=1$ 的演化 (Refining Equation 1 & 2):

我们将投影区间分割。设 $\phi_1$ 是第一步退化后的角度范围，$\Delta\phi_1$ 是丢失的角度区间：

$$S_1 = \int_{\theta=0}^{\phi_1} P(\theta) d\theta + \beta_1 \epsilon_1 = \left( \int_{0}^{\phi_0} P(\theta) d\theta - \int_{\phi_1}^{\phi_0} P(\theta) d\theta \right) + \beta_1 \epsilon_1$$

模仿 $BlurDM$ 的比例表示法 ：



$$S_1 = \frac{\phi_1}{\phi_0} S_0 - \frac{1}{\phi_0} a_1 + \beta_1 \epsilon_1$$

其中：

- 

  $a_1 = \int_{\phi_1}^{\phi_0} P(\theta) d\theta$ 定义为**角度残差 (Angle Residual)** 。

  

  

- 

  $\epsilon_1$ 为引入的高斯噪声，$\beta_1$ 为噪声缩放系数 。

  

  

#### 通用前向转换步长 $t$ (Refining Equation 3):

在任意步长 $t$ 下，从 $S_{t-1}$ 到 $S_t$ 的状态转换可以定义为 ：



$$S_t = \frac{\phi_t}{\phi_{t-1}} S_{t-1} - \frac{1}{\phi_{t-1}} a_t + \beta_t \epsilon_t$$

在这个表达式中：

- **均值漂移项**：$-\frac{1}{\phi_{t-1}} a_t$ 模拟了物理上的“信息丢失”，即角度范围的坍缩。

- 

  **物理意义**：每一扩散步 $t$ 都在系统性地剥离特定的投影角度扇区，这与 $BlurDM$ 中逐步累积模糊能量  的逻辑正好相反，但数学形式完全对称。

  

  

------

### 3. 分布特性与反向生成逻辑

根据上述推导，转换分布 $q(S_t | S_{t-1}, a_t)$ 依然服从高斯分布 ：



$$q(S_t | S_{t-1}, a_t) = \mathcal{N}(S_t; \frac{\phi_t}{\phi_{t-1}} S_{t-1} - \frac{1}{\phi_{t-1}} a_t, \beta_t^2 I)$$

#### 反向恢复 (Reverse Generation):

在反向过程中，你的模型任务是给定一个有限角正弦图 $S_T$（如只有 60° 采样），通过学习**角度残差估计器 $a^\theta$** 来重新填补丢失的投影角度 ：



$$S_{t-1} = \frac{\phi_t}{\phi_{t-1}} S_t + \frac{1}{\phi_{t-1}} a^\theta(S_t, t, B) - (\text{噪声项})$$
