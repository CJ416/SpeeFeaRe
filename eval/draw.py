import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class Snake(nn.Module):
    """
    Snake 激活函数
    
    公式：
    Snake(x) = x + (1/α) * sin²(α * x)
           = x + (1/α) * [1 - cos(2α * x)] / 2
    
    其中 α 是可学习参数，控制周期性的频率
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
    
    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x).pow(2)


def snake_function(x, alpha=1.0):
    """Snake 激活函数"""
    return x + (1.0 / alpha) * np.sin(alpha * x) ** 2


def snake_derivative(x, alpha=1.0):
    """
    Snake 函数的导数
    
    d/dx Snake(x) = 1 + sin(2α * x)
    """
    return 1 + np.sin(2 * alpha * x)


def plot_snake_functions():
    """绘制 Snake 函数及其导数"""
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Snake Activation Function Analysis', fontsize=16, fontweight='bold')
    
    # 定义 x 范围
    x = np.linspace(-3 * np.pi, 3 * np.pi, 1000)
    
    # 不同的 alpha 值
    alphas = [0.5, 1.0, 2.0, 4.0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # ==================== 子图 1: 不同 alpha 的 Snake 函数 ====================
    ax1 = axes[0, 0]
    for alpha, color in zip(alphas, colors):
        y = snake_function(x, alpha)
        ax1.plot(x, y, label=f'α = {alpha}', linewidth=2, color=color)
    
    # 添加参考线
    ax1.plot(x, x, '--', color='gray', alpha=0.5, label='y = x (identity)')
    ax1.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Snake(x)', fontsize=12)
    ax1.set_title('Snake Activation Function', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-3*np.pi, 3*np.pi])
    
    # ==================== 子图 2: Snake 函数的导数 ====================
    ax2 = axes[0, 1]
    for alpha, color in zip(alphas, colors):
        y_prime = snake_derivative(x, alpha)
        ax2.plot(x, y_prime, label=f'α = {alpha}', linewidth=2, color=color)
    
    # 添加参考线
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label="y = 1")
    ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel("Snake'(x)", fontsize=12)
    ax2.set_title('Derivative of Snake Function', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-3*np.pi, 3*np.pi])
    ax2.set_ylim([-0.5, 2.5])
    
    # ==================== 子图 3: 与其他激活函数对比 ====================
    ax3 = axes[1, 0]
    
    # Snake (alpha=1.0)
    y_snake = snake_function(x, alpha=1.0)
    ax3.plot(x, y_snake, label='Snake (α=1.0)', linewidth=2.5, color='#4ECDC4')
    
    # ReLU
    y_relu = np.maximum(0, x)
    ax3.plot(x, y_relu, label='ReLU', linewidth=2, color='#FF6B6B', linestyle='--')
    
    # Leaky ReLU
    y_leaky_relu = np.where(x > 0, x, 0.1 * x)
    ax3.plot(x, y_leaky_relu, label='Leaky ReLU', linewidth=2, color='#FFA07A', linestyle='--')
    
    # ELU
    y_elu = np.where(x > 0, x, np.exp(x) - 1)
    ax3.plot(x, y_elu, label='ELU', linewidth=2, color='#95E1D3', linestyle='--')
    
    # Identity
    ax3.plot(x, x, label='Identity', linewidth=1.5, color='gray', linestyle=':')
    
    ax3.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax3.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('f(x)', fontsize=12)
    ax3.set_title('Snake vs Other Activation Functions', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-3, 3])
    ax3.set_ylim([-2, 4])
    
    # ==================== 子图 4: 导数对比 ====================
    ax4 = axes[1, 1]
    
    # Snake derivative
    y_snake_prime = snake_derivative(x, alpha=1.0)
    ax4.plot(x, y_snake_prime, label="Snake' (α=1.0)", linewidth=2.5, color='#4ECDC4')
    
    # ReLU derivative
    y_relu_prime = np.where(x > 0, 1, 0)
    ax4.plot(x, y_relu_prime, label="ReLU'", linewidth=2, color='#FF6B6B', linestyle='--')
    
    # Leaky ReLU derivative
    y_leaky_relu_prime = np.where(x > 0, 1, 0.1)
    ax4.plot(x, y_leaky_relu_prime, label="Leaky ReLU'", linewidth=2, color='#FFA07A', linestyle='--')
    
    # ELU derivative
    y_elu_prime = np.where(x > 0, 1, np.exp(x))
    ax4.plot(x, y_elu_prime, label="ELU'", linewidth=2, color='#95E1D3', linestyle='--')
    
    ax4.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax4.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel("f'(x)", fontsize=12)
    ax4.set_title('Derivatives Comparison', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([-3, 3])
    ax4.set_ylim([-0.5, 2.5])
    
    plt.tight_layout()
    plt.savefig('snake_activation_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 图像已保存为 'snake_activation_analysis.png'")
    plt.show()


def plot_snake_properties():
    """绘制 Snake 函数的关键特性"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Snake Function Key Properties', fontsize=16, fontweight='bold')
    
    x = np.linspace(-2*np.pi, 2*np.pi, 1000)
    alpha = 1.0
    
    # ==================== 特性 1: 周期性 ====================
    ax1 = axes[0]
    y = snake_function(x, alpha)
    y_periodic_part = (1.0 / alpha) * np.sin(alpha * x) ** 2
    
    ax1.plot(x, y, label='Snake(x)', linewidth=2.5, color='#4ECDC4')
    ax1.plot(x, x, '--', label='Linear part: x', linewidth=2, color='gray', alpha=0.7)
    ax1.plot(x, y_periodic_part, ':', label='Periodic part: sin²(αx)/α', 
             linewidth=2, color='#FF6B6B')
    
    ax1.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Decomposition: Linear + Periodic', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ==================== 特性 2: 梯度振荡 ====================
    ax2 = axes[1]
    y_prime = snake_derivative(x, alpha)
    
    ax2.plot(x, y_prime, linewidth=2.5, color='#45B7D1')
    ax2.fill_between(x, 0, y_prime, alpha=0.3, color='#45B7D1')
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=1.5, 
                label='Average gradient = 1', alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    
    # 标注最大最小值
    ax2.axhline(y=2, color='green', linestyle=':', linewidth=1, 
                label='Max gradient = 2', alpha=0.7)
    ax2.axhline(y=0, color='orange', linestyle=':', linewidth=1, 
                label='Min gradient = 0', alpha=0.7)
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel("Snake'(x)", fontsize=12)
    ax2.set_title('Oscillating Gradient (α=1.0)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.2, 2.3])
    
    # ==================== 特性 3: Alpha 参数的影响 ====================
    ax3 = axes[2]
    
    alphas = [0.25, 0.5, 1.0, 2.0, 4.0]
    colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    
    for alpha, color in zip(alphas, colors_gradient):
        y = snake_function(x, alpha)
        # 计算周期
        period = 2 * np.pi / alpha
        ax3.plot(x, y, label=f'α={alpha:.2f} (T={period:.2f})', 
                linewidth=2, color=color)
    
    ax3.plot(x, x, '--', color='black', alpha=0.5, linewidth=1.5, label='y=x')
    ax3.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax3.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
    
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('Snake(x)', fontsize=12)
    ax3.set_title('Effect of α on Periodicity', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-2*np.pi, 2*np.pi])
    
    plt.tight_layout()
    plt.savefig('snake_properties.png', dpi=300, bbox_inches='tight')
    print("✅ 图像已保存为 'snake_properties.png'")
    plt.show()


def print_snake_formulas():
    """打印 Snake 函数的数学公式和性质"""
    
    print("="*70)
    print("SNAKE ACTIVATION FUNCTION - 数学公式与性质")
    print("="*70)
    
    print("\n📐 定义:")
    print("   Snake(x; α) = x + (1/α) · sin²(α·x)")
    print("               = x + (1/α) · [1 - cos(2α·x)] / 2")
    
    print("\n📊 导数:")
    print("   Snake'(x; α) = 1 + sin(2α·x)")
    
    print("\n🔑 关键性质:")
    print("   1. 恒等性: 当 α → 0 时, Snake(x) → x")
    print("   2. 周期性: 周期 T = 2π/α")
    print("   3. 连续性: 在整个实数域连续且可微")
    print("   4. 梯度范围: Snake'(x) ∈ [0, 2]")
    print("   5. 平均梯度: E[Snake'(x)] = 1")
    
    print("\n💡 优势:")
    print("   ✓ 周期性特征提取 (适合音频信号)")
    print("   ✓ 避免梯度消失 (梯度始终 ≥ 0)")
    print("   ✓ 可学习的频率参数 α")
    print("   ✓ 计算高效 (只需 sin 运算)")
    
    print("\n🎵 在音频 Codec 中的应用:")
    print("   • DAC (Descript Audio Codec) 使用 Snake 作为主要激活函数")
    print("   • 周期性特征捕捉音频的谐波结构")
    print("   • 相比 ReLU/ELU，更适合连续波形信号")
    
    print("\n⚙️ 参数 α 的影响:")
    print("   • α 小 → 周期长 → 更接近 identity")
    print("   • α 大 → 周期短 → 更强的周期性")
    print("   • 典型值: α ∈ [0.5, 2.0]")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # 设置中文字体（如果需要显示中文）
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 打印数学公式
    print_snake_formulas()
    
    # 绘制主要分析图
    print("🎨 生成 Snake 函数分析图...")
    plot_snake_functions()
    
    # 绘制特性图
    print("\n🎨 生成 Snake 函数特性图...")
    plot_snake_properties()
    
    print("\n✨ 完成！")