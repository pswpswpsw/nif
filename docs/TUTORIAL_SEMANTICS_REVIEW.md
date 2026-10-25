# 八个 PyTorch tutorial 的独立语义审计

审计日期：2026-10-25。结论是 **source/data 静态审计结论，不是数值等价、收敛等价或运行通过声明**。

审计快照：当前 `pytorch` 源码为 `6a12ec8959b0cc2a9e3a2e3296e168f727b05adc`；报告本身及下方协调者复核是后加文档，不改变该源码快照。

## 范围、基准与方法

- 原始基准：`tutorial/1..8` 的原始 `.ipynb`（含内嵌历史输出），以及 Git commit `ad9113df14590189b12068b6b3fa7c8c8d9187d4` 的 TensorFlow 库源。已确认该 Git 对象存在。
- 分别核查原始 notebook、现代 `*_colab.ipynb` / `tutorial/colab_support` 适配，以及对应 `*_pytorch.ipynb`。现代适配已经修正的行为，不倒推为原始行为。
- 四个独立只读子审计分别覆盖 T1–2、T3–4、T5–6、T7–8；汇总人另行检查实际调用的 model/layers/data/optimizer 路径，并直接复核重点表达式。
- 仅用文本、JSON/AST、Git 对象及 Python 标准库 ZIP/NPY 原数据解析。**没有导入 TensorFlow、PyTorch 或 nif，没有运行模型、训练、GPU、完整 notebook 或测试。**没有读取 parity 报告、tests、dev_tests、private/coord 结果、其他审计报告或 memory。
- 本次只创建此报告，没有修改代码、notebook、数据、阈值或测试。未找到 `D:\AGENTS.md` 或 `D:\nif\AGENTS.md`；遵循已提供的用户级 AGENTS 指示。可用 skills 中没有比直接源代码/原数据审计更匹配的流程，未强套论文或安全审计 skill。
- 下文 cell 均为 **0-based JSON cell index**。`原Git path:line` 指上述 commit；未加“原Git”的库路径指当前工作树。T1–T8 对应编号同名教程；“native”指 `_pytorch.ipynb`。

## 八行总表

| 教程 | 保留的主要任务 | 静态审计结论及边界 |
|---|---|---|
| T1 simple 1d wave | 低频行波、Swish NIF、AdaBelief 后 L-BFGS | 任务/数据/主要公式保留；GC 原意图与默认实际路线需分开；工件及色标有改编。未验证数值等价。 |
| T2 multi scale NIF | 高频行波、Swish 与 SIREN 两组拟合 | 配置、数据、scheduler、训练点重建保留；颜色尺度改变影响弱拟合的视觉解释。未验证拟合结论。 |
| T3 linear NIF | CylinderFlow 线性时空分解、面积加权重建 | 真实数据、线性表示和加权目标保留；GC 适配及时间末端索引有变化，新增训练集指标。未验证数值等价。 |
| T4 gradients | 4→5 tanh 的完整输出及 selected J/H | 原主体轴语义保留；增加解析/零导数检查及一次优化，checkpoint 与图/metrics 属不同更新时刻。 |
| T5 TFRecord | 百万行、100 个真实 TFRecord shard、两遍训练 | 核心流程保留；target singleton 维修正区别于原Git；新增在线 loss/计数/checkpoint。未验证运行。 |
| T6 regularization | baseline + 四 L2 + 四 L1、各 3000 epoch | 数据/正则目标/训练规模保留；网格命名与排序更明确，新增域内细化网格物理误差；未验证原文实验结论。 |
| T7 compression | 高频 NIF、指定 kernel pruning、部署压缩 | pruning/quantization 后端改编需注明；**gzip 大小混入 optimizer 状态**；原指标及图义存在已确认变化。 |
| T8 Sobolev | 同一高频场，有/无空间导数监督 | field + spatial derivative 目标及链式缩放保留；物理反归一修正已在现代TF适配中出现；新增指标仍需运行验证。 |

## A. 确认的语义问题与差异

### A1. T7 压缩大小基准混入训练状态（确认的 benchmark 语义问题）

- 原TF cells **12、34**：`save_model(..., include_optimizer=False)` 分别保存 `original.h5` 与 `stripped.h5`；cell **40** 比较二者及两种 TFLite 推理工件的 gzip 大小。
- native cell **2** 的 `save_checkpoint` 无条件写入 `optimizer_state`；cells **8、28** 的 `save_and_check` 因此生成包含 Adam m/v 等训练槽的 `original.pt` / `stripped.pt`。cells **29、33** 保存的 TorchScript 工件则只含推理模型。cell **37** 将四者并列列为 `gzip_bytes`。
- 后果：前两项计量“模型 + optimizer 训练状态”，后两项计量推理工件；不再对应原教程排除 optimizer 的模型压缩实验。pruned 权重为零也不意味着 Adam 槽为零，可能显著改变 gzip 大小及压缩率解释。**仅凭这些 bytes 不能归因模型压缩收益。**高置信源码事实；未生成文件，未声称实际大小或变化方向。可以保留训练 checkpoint，但应将其大小与部署模型大小明确区分；本次不实施修改。

### A2. T7 谱范数指标改为 Frobenius 指标（确认差异，改动意图 [VERIFY]）

- 原TF cells **44、47** 对 `(200,10)` 二维矩阵计算 `np.linalg.norm(error, ord=2)**2 / np.linalg.norm(unpruned, ord=2)**2`，即矩阵谱范数平方之比。
- 现代TF `_colab` 同 cells 已改为 `ord='fro'`；native cell **37** 为 `sum(error**2)/sum(unpruned**2)`，与 Frobenius 形式相同。
- 后果：从最大奇异值能量比例变成全矩阵元素能量比例；两指标可给出不同误差大小，亦可改变方案间排序。不能直接与原始 notebook 数字比较，也不能自动称作纠错或纯展示变化。高置信公式差异；未找到说明改动动机的公共注释，故动机 **[VERIFY]**。

### A3. T7 右下差值图的比较对象改变（确认差异，改动意图 [VERIFY]）

- 原TF cell **46** 右下画 `y_pred-y_pred_quant`，是 pruned 与 quantized-pruned 之间的差。
- native cell **36** 两列统一画 `compressed-y_pred_unpruned`，右下变为 quantized-pruned 相对 unpruned 的总差；左下仍是 pruned-unpruned。
- 后果：原右下可隔离量化附加影响，新右下合并 pruning 和 quantization 的影响。native cell **37** 另给 `quantization_mse_vs_pruned`，但不能据此认为两张图相同。高置信；动机 **[VERIFY]**。

除 A1 外，未在本次实际读取的主体路径中确认新的 PyTorch 科学计算错误；这不构成所有路径无错的保证。A2/A3 是确认的定义差异，不凭静态差异自行判定作者意图或数值优劣。

## B. 有明确依据的修正或后端改编

### B1. 三份原始数据及 normalization 保留

| 原始文件 | 与原Git blob的比较 | stdlib NPY核查 |
|---|---|---|
| `nif/demo/dataset/traveling_wave.npz` | 字节相同；blob `e16c0c12a2c4b5fa4ffed438579f6dce16675e8e` | float32 C-order `(2000,3)`，`[t,x,u]` |
| `nif/demo/dataset/traveling_wave_high_freq.npz` | 字节相同；blob `3487e506f8fdeb3bdffca19e12270c50affa7f1f` | float32 C-order `(2000,3)`，`[t,x,u]` |
| `nif/demo/dataset/cylinderflow.npz` | 字节相同；blob `c0e9a40cb74838fb91b2d12a79519fcd7077563d` | float64 Fortran-order `(191775,6)`，`[t,x,y,u,v,area]` |

- TravelingWave / HighFreq 当前 demo 读取及列切分与原Git一致（各文件 `:34–40` 左右）；CylinderFlow 见 `nif/demo/cylinderflow.py:34–47`。波数据为10个时间 `0,10,...,90`，每时200个空间点 `0,.005,...,.995`，时间优先、x快变。原/新 `reshape(10,200)` 与 `contourf(tt,xx,...)` 没有轴交换。
- 解析式 `exp(-1000*(x-.2-.006*t)^2)*sin(omega*(x-.2-.006*t))`：T1 `omega=4`，T2/6/7/8 `omega=400`。只读原数据核算波形最大绝对差分别约 `1.50e-9` / `1.26e-8`，属于 float32 存盘精度量级；不是运行模型所得。
- 原Git `nif/data/point_wise_data.py:64–114` 与当前 `:63–136` 保留数学公式。T1用列均值/标准差；高频和CylinderFlow用 feature中点/半范围。target 为 `(u-mean(u))/max(abs(raw_u))`，**不是简单的 u/max(abs(u))，也不保证严格落在 [-1,1]**。area 只除以 mean(area)。当前新增有限性、范围和合法权重检查，不改变这些有效原数据。
- T1/T2原TF cells **17/49、5/9** 及 native cell **2** 的颜色仍是 normalized u，轴为物理 t/x；不是新发生的单位错误。T3原 cell **20** / native **19** 对 x/y/u/v 都 `normalized*std+mean` 后绘首帧物理量。

### B2. 实际模型、生成权重布局、激活及初始化

- T1/T2 Swish：原Git `nif/model.py:178–228,303–323` 对应当前 `nif/model.py:187–195,222–228`。width30、2个shortcut hidden、latent1；生成 ShapeNet 总长度 **1951**，按所有 kernels 再所有 biases，kernel reshape `[B,in,out]`；计算 `u + swish(uW+b)`，最终线性层。当前 `_unpack` `:120–135` 保留布局。
- 多尺度 full 路径（T2/6/7/8）：原Git `nif/model.py:539–736,738–960` 对应当前 `:311–359`。实际配置均无resblock，`omega_0=30, weight_init_factor=.01`；计算 **`sin(omega_0*(x@W)+b)`，bias没有乘omega**。原Git `nif/layers/siren.py:36–62` 与当前 `:27–39` 的分段初始化公式一致：first `/input_dim`、hidden `sqrt(6/width)/omega`、final `sqrt(6/(width+width))`、bias `/width`。
- T3 last-layer：原TF cell **6** / native **6**；当前 `nif/model.py:373–397`，`phi(x)` reshape `[B,2,10]`，系数 `[B,10]`，`einsum('bol,bl->bo')+bias` 对应原Git `:1219–1269` 的 Dot。HyperLinear counts `(0,0,10)` 保留；空间 SIREN first/hidden/bottleneck 区间与原Git `nif/layers/siren.py:178–204` / 当前 `:76–99` 一致。
- 普通Dense使用转置存储 `[out,in]` 并非生成权重布局被倒置。NIF Dense 的 TruncatedNormal std `.1`、边界 `±.2` 保留；T4/T5普通Dense显式Glorot uniform和zero bias对应Keras默认。
- 初始化**分布公式**保留，不代表参数样本相同。native固定seed42、原多处未固定seed；不同RNG/抽样顺序、shuffle算法和后端算术均不支持相同训练轨迹结论。

### B3. 实际 optimizer / reduction / schedule 路径

- 共用 native cell **2** 的默认 loss 为 `mean_batch(weight * mean_output((pred-target)^2))`，没有除以 weight总和；regularization 显式加到 objective；`drop_last=False`。这是对 Keras fit/reduction 的显式后端实现，源码上不见将面积权重改成概率归一化。
- T1/T3用AdaBelief：原Git `nif/optimizers/external_optimizers.py:400–431,456–529` 对应当前 `:147–162,173–241`。实际 defaults 为 beta `.9/.999`、eps`1e-14`、rectify=True、threshold5、AMSGrad=False、wd0、无warmup。均用更新后的m计算 `(g-m)^2`，variance及sqrt分母均加eps。实际不触发 sparse、AMSGrad、warmup、weight decay、clipping；未用分支不据本审计作正确性结论。
- T2/T5/T6/T7/T8用 `ReferenceAdam(...eps=1e-7)`，当前 `nif/optimizers/reference_adam.py:9–20,94–113` 保留 `m+=(g-m)(1-beta1)`、`v+=(g²-v)(1-beta2)`、`alpha=lr*sqrt(1-beta2^t)/(1-beta1^t)`、`p-=alpha*m/(sqrt(v)+eps)` 的 epsilon-hat约定。不能将它误描述为直接替换成默认 torch.optim.Adam。T4新增的一步才直接用 torch.optim.Adam。
- T1原TF cells **45、47** / 原Git `nif/optimizers/lbfgs.py:104–119` 与 native **35、37**：200 rounds，每轮fresh optimizer最多1000 iter，20 correction pairs，gradient/x/f-relative tolerance均`1e-15`，line-search limit100。native closure额外regularization在T1配置为0。当前 `ReferenceLBFGS` `:148–209` 另实现Hager–Zhang和独立停止判据；**[VERIFY]** 本次未证明线搜索/停止在数值上与TFP一致。
- FULL步数核对：T1/T2每fit5000×4=20000，尾464；T3 600×188=112800，尾287；T5 200个shard访问×79=15800，尾16；T6每fit3000×4=12000，九组108000；T7 baseline5000×4=20000、pruning50000×4=200000；T8每fit50000×4=200000，两组400000。smoke另有缩短控制，不代表这些FULL任务完成。
- T1 schedule保留；T2 Swish最后阶段回升`5e-4`也保留，不能擅改为单调下降。T3 epochs0–199/200–399/400–599分别`1e-3/5e-4/2e-4`，原scheduler的`3e-5`分支在600epochs中不触发。T6原scheduler是`return lr`，native常数`1e-3`；T5/T7 pruning/T8也没有额外scheduler被触发。

### B4. GC：原意图、bootstrap和native分开

- T1原 cell **23**、T3原 **10** 都写 `optimizer.get_gradients=centralized_gradients_for_optimizer(optimizer)`；原Git `nif/optimizers/gtcf.py:52–77` 返回public `get_gradients` helper。这表达启用GC的意图，但 **[VERIFY]** 历史Keras fit究竟调用哪条分支未运行确认。
- T1 `bootstrap_tutorial1.py:108–130` 仅迁移legacy Optimizer等兼容项，没有通用bootstrap的tape GC补丁。native T1 cell **16** 明确 `use_gradient_centralization=False`，注释称匹配现代Colab实际fit；因此不能称作原启用意图的原样等价，也不能在未运行历史路径时直接判错。
- T3现代 `bootstrap_tutorials.py:147–169` 显式给`_get_gradients`补centralization；native cell **2/9** 在backward之后直接centralize。nn.Linear沿input轴1，自定义TF-layout kernel沿轴0，与TF `range(rank-1)`一致。其余教程本次实际配置不启用GC。

### B5. T5 target shape修正，三个基准层明确不同

- 原TF cells **18、20** 调原Git `nif/data/tfr_dataset.py:96–109`。每个列tensor `[1,N]`；feature经stack/squeeze/transpose为`[N,4]`，无权重target却直接`tf.transpose(batch_file[-3:])`，结果为 **`[N,1,3]`**。
- 现代TF `bootstrap_tutorials.py:130–145` 明确插入stack/squeeze，target改成 **`[N,3]`**。这已经超过纯依赖替换。
- native cell **21** 调当前 `nif/data/tfr_dataset.py:158–167`，各列reshape(-1)，最后axis stack，feature/target分别`[N,4]`、`[N,3]`。这是与现代TF修正相同的按行目标，**不是原Git shape原样复现**。
- 后果边界：若未经loss层额外squeeze，`[B,3]`与`[B,1,3]`普通广播可引入`[B,B,3]`跨样本误差；**[VERIFY]** 原历史Keras loss前是否另行消除了singleton没有运行确认。确认的是shape差异，不用历史loss数字倒推原目标必然怎样。

### B6. T5确实保留TFRecord与分片训练

- 原 cells **3、8、11** / native **5、10、13**：FULL生成`uniform(0,1,(1000000,7))`，前四列features `(t,x,y,z)`、后三列targets `(u,v,w)`，没有解析PDE关系、normalization或area weights；本来就是toy Dense，非NIF训练。每shard10000行，共100个。
- 当前 `nif/data/tfr_dataset.py:21–52,118–148` 写真实TFRecord长度/CRC32C framing与protobuf Example，key顺序`input_0..3,output_0..2`；一shard是**一个含七个float-list列feature的Example**，并非NPZ假冒。native markdown cell **23**纠正原markdown **22** “seven Examples”的说法。
- 原 writer 和native都整体打乱行，未拆散feature/target对应；native文件名排序（`:174`），原glob未指定排序，RNG也不同。训练读取流式，创建随机数组及NPZ转换仍先持有全数组，不能称端到端常量内存。
- 原TF cell **16** / native **17**：4→3 tanh Dense、Adam、MSE；native每次shard访问继续同一optimizer。原本也连续使用optimizer，并非新增状态连续性修复。full数据没有尾shard，但有16行尾batch。

### B7. T6坐标命名修正不等于换了测试点集合

- 原 cell **5**：`test_data=vstack([test_xx.flatten(),test_tt.flatten()]).T`；native cell **6**改`column_stack([test_tt.ravel(),test_xx.ravel()])`，注释明确“ALWAYS [parameter, coordinate] = [t,x]”。
- 两轴原本都是同样371点`[-1,1]`，所以两者是同一371²笛卡尔点集合，只是row ordering/二维数组转置。原callback cell **7** 等用`contourf(test_xx,test_tt,test_u)`，native **9**同步改`contourf(test_tt,test_xx,test_pred)`，不能单凭列名字断言旧图科学任务不同或错误。
- native **9** 新增按各feature对应的`tw.std/mean`恢复物理 t/x、求解析u、反归一prediction，再算physical MSE。这是明确的新验证展示；网格位于`t∈[0,90], x∈[0,.995]`内，是域内加密插值，不是外推。

### B8. T8物理单位修正

- 原TF cells **36、38** 将normalized prediction直接与physical解析`u_test`画在一起。现代 `_colab` cell **36** 已补`prediction*tw.std[2]+tw.mean[2]`。
- native cell **31**注释明确“Correct inverse scaling: analytic u_test is physical”，按同样式子恢复u；cell **33**纵轴明确physical u。
- 这是有公共注释支持的单位修正，native沿用现代TF适配，不能说原始比较式完全保留。高置信公式/单位判断；对最终模型误差大小未作运行判断。

### B9. T7 pruning及quantization覆盖是具体子集

- 原TF cells **4、8、18、20** 的ParameterNet为Swish，`'SIREN' in layer.name`只选最终`HyperLinearForSIREN`，原cell **20**内嵌输出也显示这一层。原Git `nif/layers/siren.py` 的`HyperLinearForSIREN.get_prunable_weights()`只返回`[self.w]`。native **5、6、16**按类选择同一层的`w`，kernel为 **`(1,1951)`**；1951个bias、此前ParameterNet层、所有动态生成ShapeNet kernel都不是独立被剪枝参数。原cell **30**“1951 weights / 195 nonzero”的历史数值不能称整个模型90%稀疏。
- 原TF **14、26**配置50000epochs、4steps/epoch、end_step200000、frequency4000、initial/final sparsity `.10/.90`。native **12、16、20**保留这些FULL配置。当前 `nif/pruning.py:34–56` 在每次training forward前step加1，符合条件时依当前参数幅值刷新threshold/mask；每个forward前赋`weight*=mask`，梯度和Adam槽不屏蔽。epoch end再施mask（`:113–121`），strip删除控制状态、不新建参数。
- native首次刷新在step4000，前3999步dense，第一次实际schedule值约14.70464%，共50次刷新，终点90%；不是step0已经施加10%mask。threshold保留所有ties，实际sparsity可偏离schedule。**[VERIFY]** 本次没有读取外部TFMOT算子源或运行wrapper，故只确认原配置与native可见行为，不声称精确step/ties等价。
- 原TF **36、40–42**用TFLite `Optimize.DEFAULT`，导出图内的真实量化算子集合未检查 **[VERIFY]**；逐点解释器输入`(1,2)`。native **33**明确后端替代：最终HyperLinear转成nn.Linear，随后dynamic int8量化所有nn.Linear。在当前配置静态可推导为 **3个** Linear：1→30输入、30→1瓶颈、1→1951最终projection。两个shortcut的`DenseLayer.w @`保持浮点（`nif/layers/mlp.py:36–51`、`nif/layers/_common.py:158–185`），generated ShapeNet weights和sine亦保持浮点。
- native **34**一次以`(2000,2)`调用restored quantized artifact；dynamic activation quantization及批大小与原逐点TFLite路线不同，不能由同后端reload自检推导跨框架量化等价。native cell **0/33**公开注明native替代，因此这是后端改编；实际算子执行/精度未运行验证。
- 原cell **38**函数虽名gzip却用ZIP_DEFLATED，native **31**才用gzip，加上HDF5/TFLite→PT/TorchScript容器变化，字节数本来就不能逐数复现。A1额外指出了训练状态混入的问题。

### B10. T8导数目标、loss和链式缩放

- 原TF cell **15** / native **16** 的target顺序均为`[u_norm, du_dt/(scale_u/scale_t), du_dx/(scale_u/scale_x)]`，features仍为`[t_norm,x_norm]`。这是`du_norm/dt_norm`和`du_norm/dx_norm`所需链式因子；常数feature/target均值的导数为0。
- 原TF **20** / native **19**的实际优化目标均为 **`field_mse + 1e-3*spatial_derivative_mse`**。`JacobianLayer(model,[0],[0,1])`给`[B,1,2]`，`jac[:,0,0]`为t，`jac[:,0,1]`为x。temporal derivative只作为metric，不被加进loss；不能因native另打印weighted_temporal_mse就称其监督了时间导数。
- 原TF **22** / native **21**的baseline只优化field MSE。两组相同50000epochs、lr`1e-4`、无GC、无配置regularizer。原Git `nif/layers/gradient.py:222–231`与当前 `:39–64,92–98`保留逐点模型output/input轴意义。
- native **24**的物理diagnostics用`[scale_u, scale_u/scale_t, scale_u/scale_x]`恢复value/derivative，mean为`[mean_u,0,0]`。这和训练target的逆变换一致；不是把归一化Jacobian误作physical Jacobian。没有运行实际导数或参数梯度比较。

### B11. 原始与现代Colab notebook的直接source差异清单

以下均逐cell JSON source比较；不将bootstrap执行结果视为已运行事实。

| 教程 | 除cell0 bootstrap外的notebook source变化 |
|---|---|
| T1 | cell4将GPU查询改为条件执行；cell37恢复前compile的Adam改legacy.Adam；cell39 load weights后新增全训练点prediction close断言。 |
| T2 | 无。 |
| T3 | 无；GC差异来自通用bootstrap库补丁。 |
| T4 | 无。 |
| T5 | cell13把`%ls`换成Path.glob文件名/字节数展示；target修正来自bootstrap库补丁。 |
| T6 | 无；测试网格顺序未在Colab notebook内改动。 |
| T7 | cells44/47谱范数改Frobenius；详见A2。 |
| T8 | cell36给超分辨prediction补物理反归一；详见B8。 |

两个bootstrap均固定原NIF commit；依赖声明为TF2.21、tf-keras2.21、TFP0.25等（各脚本`:16–28`）。通用bootstrap另外修target维度和GC hook，T1 bootstrap没有这两项。这里只核查声明/patch source，没有安装或执行它们。

## C. 额外工作量、结果范围与展示变化

### C1. T1/T2共用色标改变影响视觉解读

- 原T1 cells **17、49** `vmin=-5,vmax=5`；T2 cells **5、9** `vmin=-6,vmax=6`。native cell **2** `wave_figure`对true/pred/error分别`contourf(...,levels=50)`，移除了固定范围。
- 后果：每panel自动选择自身色域；接近常数的弱拟合prediction或很小的error也可用满色域，跨panel颜色强弱不再表示同一幅值。特别是T2教材Swish弱拟合示例，不能仅凭native图更鲜明推断拟合改善。没有修改数值，只改变观察尺度；未运行native图，不声称已见其实际视觉效果。
- T6 native **9** 则使用三个panel共用levels和`normalized u`色条，原callback各自contourf并有`vmin/vmax`；T7原 **46**固定`linspace(-1,1,50)`，native **36**每panel自动levels+colorbar。两者也不具像素/色强等价性。

### C2. T3时间末端索引改变，但首帧重建点不变

- 原TF cell **16** `diff(t,append=0)>0`得19个结束边界；native **15** `r_[flatnonzero(diff(t)!=0)+1,len(input_pred)]`得20个，纳入末帧终点191775。公共native代码没有说明动机，**意图[VERIFY]**；可确认其补全末端行为。
- 原数据20个递增时间快照，首帧10087行、末帧8927行。两式首边界均10087；原 cell **20** / native **19** 的`ti=0`重建选点相同。
- 原 cell **18** 的`diff(time_jump_index)`画18个间隔，native **17**画19个；二者都没prepend0，仍未画首帧10087的count。横轴实际ordinal index，虽label“time”。不能称所有AMR count展示已完整。
- native **13**新增phi shape `(32,2,10)`；**19**新增全191775行weighted normalized MSE及unweighted physical u/v MSE。FULL为训练点重建；SMOKE仅训练前4096行（全在首快照）却仍评估全数据，因此其metric混合训练与未训练行，不能解读为FULL训练误差。

### C3. T4新增训练，且保存模型与图/metric时刻不同

- 原TF cells **4、8、10、12、16、18** / native **5、9、11、13、17、19** 主体保留4→5 tanh、`10×4 uniform[0,1)`输入、无physical normalization；完整`y[10,5]`、J`[10,5,4]`、H`[10,5,4,4]`。
- 原Git `nif/layers/gradient.py:251–261` / 当前 `:49–100` 保留H轴`[batch, selected_output, selected_input, selected_input]`，并返回完整y；actual Dense逐点、无batch coupling，sum-output反传可给逐点导数。
- native **19**新增analytic tanh值检查、非顺序indices `y=[4,1],x=[3,0]`、linear/constant零高阶导，以及一次 `derivative_loss=mean(J²)+mean(H²)` 的Adam更新。原TF仅求导/打印shape、不训练。注释明确导数罚项需可训练，这是额外任务。
- 同cell先算J/H与expected tensors，之后`optimizer.step()`、save/reload；最后图及`jacobian_max_error/hessian_max_error/derivative_loss`仍取更新前tensor。因此checkpoint是post-step，图/metrics是pre-step。高置信状态时序事实；不将其误写为某数值断言失败。
- 当前gradient `:34–47`为常数/线性构造connected zero，`enable_grad/create_graph`保留导数loss参数梯度；这是有新增例子调用的健壮性扩展，不能称原随机tanh教程已有的验证。

### C4. T5打印MSE是在线训练平均，不是最终模型重算误差

- native cell **21**新增`rows_seen`、`shard_visits`、遍数断言、checkpoint、loss图和JSON；源码中的`mse=accumulated_loss/rows_seen`跨训练期间各更新及两遍数据，`stream_losses`为每次shard访问的在线平均。
- 原 cell **20**逐shard fit打印loss；两边这些loss均随模型更新而产生。native新汇总不是最终模型对全部百万行重新推理的MSE，也不是held-out验证。新断言存在不表示本次已经运行通过。

### C5. T6九组正则实验保留，新增误差单位要分清

- 原TF cells **7–34**对应native **8–27**：baseline和L2/L1各`1e-5,1e-4,1e-3,1e-2`，每次fresh model/Adam。原Git `nif/model.py:109–114,668–733` / 当前 `:138–144,255–265`对parameter net所有kernel和bias求sum正则，加进loss；不是optimizer weight decay。L2优先规则保留，但这些配置各只给一种。无activity/Jacobian正则。
- native **9**的`training_mse`是最终2000训练行normalized-u MSE；`physical_superresolution_mse`是371²域内细化点physical-u MSE，单位和采样均不同；`regularization_loss`用64行调用但此处只依赖参数，不受这64行选取影响。
- 原每100epoch画训练/预测/细化图；native每100打印历史、每1000/final保存latest，每fit结束才画最终图，观察频率改变。原文字关于“L1/L2不能消除wiggles”等实验结论仍保留，**[VERIFY]** 当前PyTorch full九组是否实际复现。

### C6. checkpoint、derived views、history都是不同后端工件

- native共用cell **2**保存`model_state/optimizer_state/config/epoch/framework/smoke`；`save_and_check`重建模型并比较示例预测，只加载model_state，**不恢复optimizer并验证继续训练**。保存的optimizer state存在，不等于resume行为已被证明；本次连预测断言也未执行。
- T1原 **29**的parameter→latent SavedModel，native **21**只导出view `state_dict`，需代码/结构重建；不是相同部署接口。native **19、23、25**新增latent全训练时间曲线和derived submodel训练行检查；**38**新增训练MSE，没有额外held-out点。
- T1原 **35–39** load最终weights，native **30**保存当前模型再重载64训练行；T2 native **7、9**新增同类恢复核查。native latest文件覆盖旧checkpoint，原保留epoch目录；TensorBoard/周期预测图改为JSON/最终图。history原按100epoch抽样，native逐epoch记录，曲线横轴密度不同。
- T1 L-BFGS native **37**横轴写closure evaluation；原Git `nif/optimizers/lbfgs.py:76–85,123–126`历史本来也是objective调用次数，并非accepted optimizer update次数。

### C7. T7压缩诊断点与T8空间细化点不是同一种“验证”

- T7原 **8、44–47** / native **8、26、34、37**按normalized `t10 × x200`重建训练域网格，reshape`(200,10)`配`meshgrid(t,x)`；这是x-major展示排序，与波训练数组time-major不同，但feature和prediction配对一致。比较目标是**未剪枝模型prediction**，不是解析physical ground truth；compression fidelity不能替代原模型拟合准确度。
- T8原 **27、31、33**逐一展示10个训练time snapshot的归一化量；native **24**只画snapshot0和5的physical value/derivative六子图，**26、33**新增全2000训练点component MSE。展示覆盖缩减且单位改变，但数字诊断覆盖全部训练点。
- T8原 **35** / native **30**同样用10个训练时间、每时800空间点`0,.00125,...,.99875`。每时包含200个原训练空间点、597个训练范围内新点，以及 **3个超过训练最大x=.995的点** `.99625,.9975,.99875`。因此8000点的整体MSE主要是域内空间细化并含30个轻微空间外推点；没有时间外推，亦非独立随机validation split。
- native **31/33**反归一后在上述8000点和解析u比较，且仍展示全部10个time；native **23/24/26**则是2000个训练点的导数/value诊断。必须分别解释。

## D. 尚未验证与解释限制

1. **[VERIFY] 所有运行结果**：没有执行检查点恢复、行数断言、analytic assertions、pruning/quantized kernel或训练；存在代码不等于通过。没有用浮点误差或未运行假设断言bug。
2. **[VERIFY] 后端数值等价**：同公式、初始化分布、batch与epoch不代表相同权重、轨迹或拟合质量。尤其T1的TFP/Hager–Zhang停止行为、历史/现代Keras的GC分支与T5历史loss的singleton处理未运行确认。
3. 原notebook内嵌输出是历史证据；native八个notebook的code execution_count均空且没有内嵌outputs。不能把旧图/旧数值或新源码中的assert当成当前native运行结果。T3历史untracked-bias warning也不能被忽略后宣称历史状态与当前Git库已验证一致。
4. T1/T2/T3的FULL重建及最终MSE是训练点；T6新增grid是域内插值；T7 comparison以未剪枝prediction为目标，T8细化网格还含30个轻微空间外推点（见C7）。不能把“更密”“重建”“恢复一致性”自动称作泛化验证。
5. 本审计只评价教程实际调用配置。未用模型分支、任意batch-coupled模型、其他precision政策、sparse optimizer、全流程峰值内存和部署硬件性能，不在已确认范围内。

## E. 协调者的独立复核与改动出处

以下是独立报告完成后，协调者新增的复核；不属于上述四组审计的执行记录。独立报告原稿已按其原始 SHA256 `4f229d5a8fae6871d0b87758903dd5582776205e4720c8e45635aa03ec94d0ff` 另行保留。

- 直接读取历史 Git 对象确认：提交 `f8f9d31`（2026-10-21，`Fix Colab compatibility for all eight tutorials`）中的 T7 `_colab` cells 44/47 已使用 `ord='fro'`；提交 `eac4524`（2026-10-22，`Port NIF and eight tutorials to native PyTorch`）中的 native cell 37 已使用平方元素和之比。A2 的差异在兼容版及初次移植中均已存在，不是本次看图时新增。Git 历史能确定改动阶段，不能据提交标题证明改动动机。
- 协调者仅用 NumPy 做了一个合成矩阵算术检查：基准矩阵 `I₂`，误差 A=`diag(1,0)`、B=`0.8 I₂`。原谱范数比 A=1、B=0.64；native 平方元素和比 A=0.5、B=0.64，排序反转。这说明指标定义变化可能影响结论；它不是教程模型的实测误差，也未运行任何 ML 框架。
- T7 A1/A3 及 T4 C3 的实际 notebook 表达式已由协调者再次直接读取确认。此次只记录审查结果，不实施修复。后续若修复，应先对原定义写出失败的复现，再验证最小修改，保留各类输出的含义和来源。
