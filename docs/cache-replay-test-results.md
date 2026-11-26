# SWE 缓存回放：Test 结果

114 个实例、342 条 workflow、3,552 次调用；并发 16/32/64 × 缓存预算比例 0.25/0.50/0.75 × 入场 seed 0/1/2，每种方法 27 次，共 216 次回放。

| 方法 | 核心区别 | 平均重算 tokens ↓ | Token 命中率 ↑ | 比 LRU 少重算 ↑ | 比同组 step 少重算 ↑ |
|---|---|---:|---:|---:|---:|
| LRU | 按最近访问时间淘汰 | 39,239,138 | 26.29% | 0.00% | — |
| LAE | 优先淘汰使用者均已结束的节点 | 33,926,408 | 32.74% | 13.54% | — |
| step + Markov | Markov 角色概率＋步数折扣 | 24,476,988 | 44.21% | 37.62% | — |
| ETA + Markov | 同组 step＋真实未来时间权重 | 23,105,618 | 45.88% | 41.12% | 5.60% |
| step + predictor | 学习式角色概率＋步数折扣 | 21,295,867 | 48.08% | 45.73% | — |
| ETA + predictor | 同组 step＋真实未来时间权重 | 20,332,160 | 49.25% | 48.18% | 4.53% |
| 角色／时间 oracle | 同一打分公式，角色和时间均取真值 | 20,104,828 | 49.52% | 48.76% | — |
| 离线下一次访问 | 查看完整未来前缀访问，优先淘汰最晚再用节点 | 14,529,161 | 56.29% | 62.97% | — |

口径：重算量取 27 次回放均值；命中率按 token 汇总。下降比例＝（对照重算总量－该方法重算总量）÷对照重算总量；正值表示改善。

ETA + predictor 比 LRU 少重算 **48.18%**，其中时间权重相对同组 step 的增益为 **4.53%**；换成离线下一次访问还能再少重算 **28.54%**。时间权重并非各容量档都有效：predictor 组在 0.25/0.50/0.75 三档分别为 +6.72% / +6.50% / −2.53%。

边界：这是当前有效 test 子集（available_subset、development_only）的离线缓存模拟。ETA 使用真实未来时间；离线下一次访问是完整未来信息参考，不能视为严格最优。表中收益仅指重算量，不代表在线延迟或解题率提升。

## 各组公式（简述）

**共同打分。** 对候选缓存节点 \(c\)，保留分数为：

$$
S(P,U;c,t)=
\sum_{w\in\mathcal W_{\mathrm{active}}(c,t)}
\sum_{k=1}^{K}\gamma^{k-1}s_w^{(k)}(P)
\left(\sum_{a\in O_w(c)}P_w^{(k)}[a]\right)U_w^{(k)}(t),
\qquad
s_w^{(k)}(P)=\prod_{j=1}^{k-1}\left(1-P_w^{(j)}[\mathrm{END}]\right).
$$

\(\mathcal W_{\mathrm{active}}(c,t)\) 只包含当前仍活跃、且已被记录为节点 \(c\) 使用者的 workflow；\(O_w(c)\) 为其中已记录的使用角色，\(P_w^{(k)}[a]\) 为未来第 \(k\) 步的角色概率。\(s_w^{(1)}=1\)，后续存活权重表示此前尚未 END；各 workflow 贡献直接求和，END 不贡献角色分数。固定 \(K=3\)、\(\gamma=0.5\)、\(\tau=15.1266\) 秒。

**step／ETA／oracle 五组。** 各组只替换共同公式中的角色概率和时间权重：

$$
\begin{aligned}
S_{\mathrm{step+Markov}}&=S(P_{\mathrm{Markov}},1;c,t),\\
S_{\mathrm{ETA+Markov}}&=S(P_{\mathrm{Markov}},U_{\mathrm{true}};c,t),\\
S_{\mathrm{step+predictor}}&=S(P_{\mathrm{pred}},1;c,t),\\
S_{\mathrm{ETA+predictor}}&=S(P_{\mathrm{pred}},U_{\mathrm{true}};c,t),\\
S_{\mathrm{oracle}}&=S(P_{\mathrm{true}},U_{\mathrm{true}};c,t),
\end{aligned}
$$

$$
U_{\mathrm{true},w}^{(k)}(t)=
\exp\!\left(-\frac{\max(t_{\mathrm{start},w}^{(k)}-t,0)}{\tau}\right),
\qquad
P_{\mathrm{true},w}^{(k)}[a]=\mathbf1\{a=a_{\mathrm{true},w}^{(k)}\}.
$$

Markov 与 predictor 分别提供转移模型和学习式模型的角色概率；step 不加时间权重，ETA 在同组 step 上加入“越快使用，权重越高”的因子；oracle 再把角色换成真值。未来不足 \(K\) 步时，ETA 时间表沿用原实现，以 train 中位步长补齐。五组均先淘汰使用者全部结束的 retired 节点（先使用者少，再按 LRU），其余节点按分数从低到高淘汰，平局按 LRU。

**LRU／LAE。** 令 \(\mathcal E(t)\) 为满足共同叶节点与锁定约束的可淘汰集合，\(\ell_c(t)\) 为最近访问／释放时刻，\(V(c)\) 为已记录的使用者集合：

$$
c^*_{\mathrm{LRU}}=\arg\min_{c\in\mathcal E(t)}\ell_c(t),
\qquad
c^*_{\mathrm{LAE}}=\arg\min_{c\in\mathcal E(t)}
\begin{cases}
(0,|V(c)|,\ell_c(t)),&c\text{ 为 retired},\\
(1,0,\ell_c(t)),&\text{否则}.
\end{cases}
$$

LAE 在 LRU 基础上增加生命周期优先级；两组均不使用角色或未来时间预测。

**离线下一次访问。** 令 \(q\) 为已处理调用数，\(\mathcal P_j\) 为第 \(j\) 次调用的输入前缀集合：

$$
N_c(q)=\min\{j\ge q:\operatorname{prefix}(c)\in\mathcal P_j\},
\qquad
c^*_{\mathrm{next}}=\arg\min_{c\in\mathcal E(t)}
\bigl(-N_c(q),\ell_c(t),\operatorname{id}(c)\bigr).
$$

无后续访问时 \(N_c=+\infty\)。该组查看完整未来访问序列，先淘汰永不再用、再淘汰下一次使用最远的节点；它不受 \(K=3\) 的预测窗口限制，仍遵守共同叶节点和锁定约束。

原始回放结果与实验日志未随仓库发布。
