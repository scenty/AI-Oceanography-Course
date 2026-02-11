
"""
The Culling Line Theory of Intelligence in the AI Era
AI时代个人智能斩杀线理论 - 完整实现

Author: AI Assistant
Date: 2026-02-04
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

class CullingLineTheory:
    """
    智能斩杀线理论核心类
    
    核心公式:
    1. 学习曲线: S(t) = L / (1 + exp(-k(t-t0)))
    2. 斩杀条件: L < S_AI 或 t_cross > T_threshold
    3. 生存边界: (k > k_75%) 或 (t0 < t0_50% 且 L > 2*S_AI)
    """
    
    def __init__(self, n_population=500, seed=42):
        """
        初始化群体参数（正态分布）
        
        Parameters:
        -----------
        n_population : int
            群体规模
        seed : int
            随机种子
        """
        np.random.seed(seed)
        self.N = n_population
        
        # 参数1: 技能上限 L ~ N(70, 13^2), 99分位≈98.5（最强人类线）
        self.L = np.clip(np.random.normal(70, 13, n_population), 30, 110)
        
        # 参数2: 学习斜率 k ~ N(0.08, 0.03^2)（聪明程度）
        self.k = np.clip(np.random.normal(0.08, 0.03, n_population), 0.02, 0.20)
        
        # 参数3: 半饱和时间 t0 ~ N(50, 15^2)（进入快速增长期时点）
        self.t0 = np.clip(np.random.normal(50, 15, n_population), 20, 90)
        
        # 预计算分位数
        self.k_75 = np.percentile(self.k, 75)
        self.k_50 = np.percentile(self.k, 50)
        self.t0_50 = np.percentile(self.t0, 50)
        self.L_99 = np.percentile(self.L, 99)  # 最强人类线
        
    def learning_curve(self, t, L=None, k=None, t0=None):
        """
        S型三阶段学习曲线
        
        S(t) = L / (1 + exp(-k(t-t0)))
        
        Stages:
        1. Slow start (t << t0): S < L/2
        2. Rapid growth (t ≈ t0): max slope = k*L/4  
        3. Plateau (t >> t0): S -> L
        """
        if L is None: L = self.L
        if k is None: k = self.k
        if t0 is None: t0 = self.t0
        
        t = np.array(t)
        return L / (1 + np.exp(-k * (t - t0)))
    
    def time_to_surpass(self, L, k, t0, S_ai):
        """
        计算技能首次超越AI的时间
        
        解方程: L / (1 + exp(-k(t-t0))) = S_ai
        得: t = t0 - ln(L/S_ai - 1)/k
        
        Returns:
        --------
        float : 超越时间（若无法超越返回inf）
        """
        if L <= S_ai:
            return np.inf  # 绝对斩杀：上限不足
        try:
            t_cross = t0 - np.log(L/S_ai - 1) / k
            return max(0, t_cross)
        except:
            return np.inf
    
    def classify_individual(self, L, k, t0, S_ai, time_threshold=80):
        """
        个体生存分类
        
        Returns:
        --------
        str : 'Culled', 'Smart', 'Durable', or 'Normal'
        """
        t_cross = self.time_to_surpass(L, k, t0, S_ai)
        
        # 斩杀判定
        if t_cross == np.inf or t_cross > time_threshold:
            return 'Culled'
        
        # Smart: 高学习斜率（前25%）
        if k > self.k_75:
            return 'Smart'
        
        # Durable: 早进入快速增长期（前50%）
        if t0 < self.t0_50:
            return 'Durable'
        
        return 'Normal'
    
    def calculate_culling_rate(self, S_ai, time_threshold=80):
        """
        计算给定AI水平下的斩杀率
        
        Parameters:
        -----------
        S_ai : float
            AI能力水平
        time_threshold : int
            心理忍耐时间阈值（默认80）
            
        Returns:
        --------
        dict : 包含斩杀率、各类生存者比例的字典
        """
        culled = 0
        smart = 0
        durable = 0
        normal = 0
        
        for i in range(self.N):
            category = self.classify_individual(
                self.L[i], self.k[i], self.t0[i], S_ai, time_threshold
            )
            if category == 'Culled':
                culled += 1
            elif category == 'Smart':
                smart += 1
            elif category == 'Durable':
                durable += 1
            else:
                normal += 1
        
        return {
            'S_ai': S_ai,
            'culling_rate': culled / self.N * 100,
            'smart_rate': smart / self.N * 100,
            'durable_rate': durable / self.N * 100,
            'normal_rate': normal / self.N * 100,
            'survival_rate': (smart + durable + normal) / self.N * 100
        }
    
    def continuous_culling_curve(self, S_ai_range=None):
        """
        计算连续斩杀率曲线
        
        Returns:
        --------
        dict : 包含AI范围和各率数组
        """
        if S_ai_range is None:
            S_ai_range = np.linspace(20, 110, 91)
        
        results = {
            'S_ai': S_ai_range,
            'culling': [],
            'smart': [],
            'durable': [],
            'normal': []
        }
        
        for S_ai in S_ai_range:
            stats = self.calculate_culling_rate(S_ai)
            results['culling'].append(stats['culling_rate'])
            results['smart'].append(stats['smart_rate'])
            results['durable'].append(stats['durable_rate'])
            results['normal'].append(stats['normal_rate'])
        
        return {k: np.array(v) for k, v in results.items()}
    
    def plot_three_stage_model(self, save_path='fig1_three_stage.png'):
        """绘制图1: 三阶段模型与斩杀机制"""
        t = np.linspace(0, 100, 200)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Three trajectories
        ax1 = axes[0]
        traj_smart = self.learning_curve(t, 110, 0.15, 35)
        traj_durable = self.learning_curve(t, 105, 0.08, 30)
        traj_culled = self.learning_curve(t, 60, 0.04, 70)
        
        ax1.plot(t, traj_smart, 'b-', linewidth=3, label='Smart (High k)', alpha=0.8)
        ax1.plot(t, traj_durable, 'g-', linewidth=3, label='Durable (Early t₀)', alpha=0.8)
        ax1.plot(t, traj_culled, 'r--', linewidth=2.5, label='Culled (Low k, Late t₀)', alpha=0.8)
        
        S_ai = 50
        ax1.axhline(y=S_ai, color='orange', linestyle='--', linewidth=2.5, 
                   label=f'AI Culling Line (S_AI={S_ai})')
        ax1.fill_between(t, 0, S_ai, alpha=0.1, color='red', label='Culling Zone')
        
        ax1.axvspan(0, 25, alpha=0.1, color='gray', label='Stage 1: Slow Start')
        ax1.axvspan(25, 45, alpha=0.1, color='yellow', label='Stage 2: Rapid Growth')
        ax1.axvspan(45, 100, alpha=0.1, color='lightblue', label='Stage 3: Plateau')
        
        ax1.set_xlabel('Practice Time (t)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Skill Level S(t)', fontsize=12, fontweight='bold')
        ax1.set_title('Three-Stage Learning Curve & Culling Mechanism', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 120)
        
        # Right: Learning rate (derivative)
        ax2 = axes[1]
        def derivative(t, L, k, t0):
            s = self.learning_curve(t, L, k, t0)
            return k * s * (1 - s/L)
        
        ax2.plot(t, derivative(t, 110, 0.15, 35), 'b-', linewidth=2, label='Smart', alpha=0.8)
        ax2.plot(t, derivative(t, 105, 0.08, 30), 'g-', linewidth=2, label='Durable', alpha=0.8)
        ax2.plot(t, derivative(t, 60, 0.04, 70), 'r--', linewidth=2, label='Culled', alpha=0.8)
        ax2.axvspan(25, 45, alpha=0.2, color='yellow', label='Window of Opportunity')
        
        ax2.set_xlabel('Practice Time (t)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Learning Rate dS/dt', fontsize=12, fontweight='bold')
        ax2.set_title('Learning Rate Dynamics', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"Figure 1 saved to {save_path}")
    
    def plot_continuous_culling(self, save_path='fig2_continuous_culling.png'):
        """绘制图2: 连续斩杀率曲线"""
        curves = self.continuous_culling_curve()
        S_ai_range = curves['S_ai']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Main culling curve
        ax1 = axes[0]
        ax1.plot(S_ai_range, curves['culling'], 'r-', linewidth=4, label='Culling Rate', zorder=5)
        ax1.fill_between(S_ai_range, 0, curves['culling'], alpha=0.2, color='red')
        
        # Mark prototypes
        prototypes = [
            (34, 'Odor Detection\\n(AI Weak)', '#27AE60'),
            (58, 'Programming\\n(AI>Average)', '#F39C12'),
            (95, 'Go (Weiqi)\\n(AI>Top Human)', '#E74C3C')
        ]
        
        for s_ai, label, color in prototypes:
            idx = int(s_ai - 20)
            if 0 <= idx < len(curves['culling']):
                rate = curves['culling'][idx]
                ax1.plot(s_ai, rate, 'o', markersize=14, color=color, 
                        markeredgecolor='black', markeredgewidth=2, zorder=10)
                ax1.annotate(f'{label}\\n{rate:.1f}%', 
                            xy=(s_ai, rate), xytext=(s_ai-12, rate+8),
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))
        
        ax1.axhline(y=50, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Social Crisis (50%)')
        ax1.axhline(y=90, color='darkred', linestyle=':', linewidth=2, alpha=0.7, label='Caste Line (90%)')
        ax1.axvline(x=self.L_99, color='black', linestyle='--', linewidth=2, 
                   label=f'Strongest Human (99th={self.L_99:.1f})')
        
        ax1.set_xlabel('AI Capability Level (S_AI)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Culling Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Continuous Culling Rate vs AI Capability', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.set_xlim(20, 110)
        ax1.set_ylim(-5, 105)
        
        # Right: Stacked composition
        ax2 = axes[1]
        other = 100 - curves['culling'] - curves['smart'] - curves['durable']
        other = np.maximum(other, 0)
        
        ax2.stackplot(S_ai_range, curves['smart'], curves['durable'], other, curves['culling'],
                     labels=['Smart Survivors', 'Durable Survivors', 'Normal Survivors', 'Culled'],
                     colors=['#3498DB', '#27AE60', '#95A5A6', '#E74C3C'], alpha=0.8)
        
        ax2.set_xlabel('AI Capability Level (S_AI)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Population Composition (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Population Structure Evolution', fontsize=13, fontweight='bold')
        ax2.legend(loc='center right', fontsize=9)
        ax2.set_xlim(20, 110)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"Figure 2 saved to {save_path}")

# Main execution
if __name__ == "__main__":
    # Initialize theory
    theory = CullingLineTheory(n_population=500)
    
    print("="*60)
    print("THE CULLING LINE THEORY - Analysis Results")
    print("="*60)
    
    # Calculate for three prototypes
    prototypes = [
        ("Odor Detection (AI Weak)", 34),
        ("Programming (AI>Average)", 58),
        ("Go/Weiqi (AI>Top Human)", 95)
    ]
    
    for name, S_ai in prototypes:
        stats = theory.calculate_culling_rate(S_ai)
        print(f"\\n{name}:")
        print(f"  AI Level: {S_ai}")
        print(f"  Culling Rate: {stats['culling_rate']:.1f}%")
        print(f"  Smart Survivors: {stats['smart_rate']:.1f}%")
        print(f"  Durable Survivors: {stats['durable_rate']:.1f}%")
        print(f"  Total Survival: {stats['survival_rate']:.1f}%")
    
    print("\\n" + "="*60)
    print("Generating figures...")
    print("="*60)
    
    # Generate figures
    theory.plot_three_stage_model()
    theory.plot_continuous_culling()
    
    print("\\nAnalysis complete!")
'''

# Save to file
with open('/mnt/kimi/output/culling_line_theory.py', 'w', encoding='utf-8') as f:
    f.write(complete_code)

print("Python code saved to: /mnt/kimi/output/culling_line_theory.py")
print("\\nFile size:", len(complete_code), "characters")
print("\\nYou can copy this code and save as 'culling_line_theory.py'")
'''
