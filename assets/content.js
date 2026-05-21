window.siteContent = {
  author: "MomenT",
  posts: [
    {
      slug: "diffusion-model-foundations",
      title: "Diffusion Model 基础理论：从生成建模到 DDPM、Score 与 Flow Matching",
      date: "2026-05-21",
      category: "学习笔记",
      summary:
        "一篇面向零生成模型理论基础读者的 Diffusion Model 长文。我们从概率建模和高斯分布讲起，完整推导前向加噪、后验均值、DDPM 训练目标、score matching、SDE/ODE、DDIM 与 Flow Matching 的关系。",
      tags: [
        "Diffusion Model",
        "Generative Model",
        "Machine Learning",
        "Probability",
        "DDPM",
        "Score Matching",
        "Flow Matching",
        "SDE",
        "ODE"
      ],
      cover: "cover-b",
      body: String.raw`
        <p class="lead">这篇文章的目标是：假设读者没有系统学过生成模型，也不默认你已经熟悉 VAE、GAN、score matching 或随机微分方程。我们从“生成模型到底想学什么”开始，逐步建立 Diffusion Model 的数学对象、训练目标和采样方法。读完后，你应该能带着清楚的问题去读 DDPM、DDIM、score-based SDE、flow matching 这些论文，而不是只记住几个公式。</p>

        <section class="article-section">
          <h2>0. 阅读路线：为什么一篇文章要讲这么长？</h2>
          <p>Diffusion Model 容易被讲得很玄：从噪声生成图片、U-Net 预测噪声、score 是梯度、SDE 可以反向走、Flow Matching 和 Diffusion 等价。每一句都对，但如果缺少中间层，初学者会觉得这些概念像从天上掉下来。</p>
          <p>所以本文采用一条比较慢、但更稳的路线：</p>
          <ol>
            <li>先讲生成模型的目标：我们到底要拟合什么分布。</li>
            <li>再讲为什么高斯噪声是一个好的起点：因为它简单、可采样、可计算。</li>
            <li>然后推导 DDPM 的前向过程：为什么 \(q(x_t|x_0)\) 有闭式形式。</li>
            <li>再推导反向后验：为什么 \(q(x_{t-1}|x_t,x_0)\) 是高斯，均值和方差从哪里来。</li>
            <li>接着解释训练目标：为什么预测噪声 \(\epsilon\) 会变成一个简单的 MSE。</li>
            <li>最后把 DDPM、score matching、SDE/ODE、DDIM、flow matching 连起来。</li>
          </ol>
          <p>本文会尽量把每一步的“为什么”说清楚。公式不是为了显得高级，而是为了把直觉固定下来：你知道每个符号是什么意思，才知道论文里的省略步骤在省略什么。</p>

          <figure class="source-figure">
            <img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png" alt="生成模型类型对比图" />
            <figcaption>参考图：Lilian Weng 对 GAN、VAE、Flow-based model 和 Diffusion model 的结构对比。本文会围绕最下面的 Diffusion 路线展开。图片来源：Lilian Weng, What are Diffusion Models?</figcaption>
          </figure>

          <div class="paper-note">本文会引用参考资料中的少量公开图示，并在图注中标明来源；其余流程图、推导框架和交互式可视化是本站为学习笔记重新组织的版本。读理论时不要只看图，重点是知道图里的每条箭头在公式里对应什么。</div>

          <h3>本文符号表</h3>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>符号</th><th>含义</th><th>第一次见到时应该怎么想</th></tr>
              </thead>
              <tbody>
                <tr><td>\(x_0\)</td><td>真实数据样本</td><td>一张真实图片，或一个真实数据向量。</td></tr>
                <tr><td>\(x_t\)</td><td>第 \(t\) 个噪声等级下的样本</td><td>它不是新图片，而是由 \(x_0\) 加噪得到的中间状态。</td></tr>
                <tr><td>\(\epsilon\)</td><td>标准高斯噪声</td><td>训练时我们亲手采样，所以知道答案；模型要学会把它预测出来。</td></tr>
                <tr><td>\(\beta_t\)</td><td>第 \(t\) 步加噪强度</td><td>每一步往图像里塞多少新噪声。</td></tr>
                <tr><td>\(\alpha_t=1-\beta_t\)</td><td>第 \(t\) 步保留的信号比例</td><td>为了让尺度稳定，保留信号和加入噪声要一起设计。</td></tr>
                <tr><td>\(\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s\)</td><td>从第 0 步到第 \(t\) 步累计保留的信号比例</td><td>它控制 \(x_t\) 里还剩多少 \(x_0\) 的影子。</td></tr>
                <tr><td>\(q\)</td><td>固定的前向加噪分布</td><td>不是神经网络学出来的，而是我们人为规定的。</td></tr>
                <tr><td>\(p_\theta\)</td><td>模型学习的反向生成分布</td><td>真正要训练的东西，用参数 \(\theta\) 表示。</td></tr>
                <tr><td>\(\nabla_x\log p_t(x)\)</td><td>score</td><td>当前噪声等级下，往哪里移动会更像数据。</td></tr>
                <tr><td>\(v_\theta(x,t)\)</td><td>Flow Matching 的速度场</td><td>把样本看成粒子，速度场告诉粒子如何从噪声流向数据。</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="article-section">
          <h2>1. 生成模型到底在学什么？</h2>
          <p>我们先从一个最普通的问题开始：什么叫“生成一张图片”？在数学上，一张图片可以看成一个很长的向量。假设图片大小是 \(H \times W\)，有 3 个颜色通道，那么它可以表示为：</p>
          <div class="equation">\[
            x \in \mathbb{R}^{H \times W \times 3}.
          \]</div>
          <p>如果我们把它拉平成一维向量，也可以写成 \(x \in \mathbb{R}^d\)。真实世界中所有“合理图片”并不是均匀分布在整个 \(\mathbb{R}^d\) 中。随机采样一个像素向量，大概率只是花屏噪声。真正像图片的点，只占据高维空间中非常小、非常有结构的一部分。</p>
          <p>我们把真实数据背后的未知分布记作：</p>
          <div class="equation">\[
            x \sim p_{\mathrm{data}}(x).
          \]</div>
          <p>训练集就是从这个分布里抽出来的一些样本。生成模型的目标，是构造一个参数化分布 \(p_\theta(x)\)，让它尽量接近 \(p_{\mathrm{data}}(x)\)。如果模型足够好，我们从 \(p_\theta(x)\) 采样，就能得到看起来像真实数据的新样本。</p>
          <p>这里有两个困难：</p>
          <ol>
            <li><strong>真实数据分布复杂。</strong>图片分布包含边缘、纹理、物体结构、语义关系，远不是一个简单高斯能描述的。</li>
            <li><strong>高维概率密度难以直接写出。</strong>我们很难直接给每张图片一个可归一化的概率密度，并且还能高效采样。</li>
          </ol>
          <p>不同生成模型在这里走了不同路线。GAN 避开显式概率密度，训练一个生成器和判别器对抗；VAE 引入潜变量并最大化变分下界；Flow 模型构造可逆变换，让密度可以用变量替换公式精确计算。Diffusion Model 的思路是：不直接从复杂分布采样，而是先把复杂分布慢慢破坏成简单高斯，再学习反向修复。</p>
        </section>

        <section class="article-section">
          <h2>2. 为什么噪声分布适合作为生成起点？</h2>
          <p>标准高斯分布 \( \mathcal{N}(0,I) \) 有三个很重要的优点。</p>
          <p>第一，它非常容易采样。我们可以用随机数生成器直接得到 \(\epsilon \sim \mathcal{N}(0,I)\)。</p>
          <p>第二，它的概率密度形式简单：</p>
          <div class="equation">\[
            p(\epsilon)=\frac{1}{(2\pi)^{d/2}}\exp\left(-\frac{1}{2}\|\epsilon\|^2\right).
          \]</div>
          <p>第三，高斯分布在线性变换下仍然是高斯。这个性质会让 Diffusion 的前向过程可以被精确推导。比如如果 \(z\sim \mathcal{N}(0,I)\)，那么 \(az+b\) 也是高斯，均值是 \(b\)，方差是 \(a^2I\)。</p>
          <p>Diffusion 的基本想法可以用一句话概括：</p>
          <div class="insight-box">先设计一条容易计算的前向路径，把真实数据分布逐渐变成高斯噪声；然后训练模型学习这条路径的反方向。</div>
          <p>注意这里的“前向路径”不是自然发生的物理过程，而是我们人为设计的训练工具。它的作用是给生成问题搭一座桥：桥的一端是真实数据，另一端是容易采样的高斯噪声。</p>
        </section>

        <section class="article-section">
          <h2>2.5 高斯工具箱：后面所有推导都靠它</h2>
          <p>很多 Diffusion 推导看起来复杂，其实反复使用三条高斯事实。先把它们讲清楚，后面公式就不会像魔法。</p>

          <h3>事实一：高斯的线性变换仍然是高斯</h3>
          <p>如果 \(z\sim\mathcal{N}(0,I)\)，令 \(x=az+b\)，其中 \(a\) 是标量，\(b\) 是向量，那么：</p>
          <div class="equation">\[
            x\sim \mathcal{N}(b,a^2I).
          \]</div>
          <p>为什么均值是 \(b\)？因为 \(\mathbb{E}[x]=a\mathbb{E}[z]+b=b\)。为什么方差是 \(a^2I\)？因为方差对缩放的响应是平方：</p>
          <div class="equation">\[
            \mathrm{Var}(az+b)=a^2\mathrm{Var}(z)=a^2I.
          \]</div>
          <p>这条事实解释了为什么 DDPM 可以把“从高斯采样”写成“均值 + 标准差乘噪声”。</p>

          <h3>事实二：独立高斯相加，方差相加</h3>
          <p>如果 \(\epsilon_1,\epsilon_2\sim\mathcal{N}(0,I)\) 且互相独立，那么：</p>
          <div class="equation">\[
            a\epsilon_1+b\epsilon_2\sim\mathcal{N}(0,(a^2+b^2)I).
          \]</div>
          <p>这里的“独立”很关键。独立意味着没有协方差交叉项，因此总方差就是每部分方差的和。DDPM 的闭式加噪公式正是靠这个性质，把很多步噪声合并成一份等价噪声。</p>

          <h3>事实三：两个高斯密度相乘，仍然得到一个未归一化高斯</h3>
          <p>一维情形最容易看。设：</p>
          <div class="equation">\[
            p_1(x)\propto \exp\left[-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right],
            \qquad
            p_2(x)\propto \exp\left[-\frac{(x-\mu_2)^2}{2\sigma_2^2}\right].
          \]</div>
          <p>相乘后指数相加：</p>
          <div class="equation">\[
            p_1(x)p_2(x)
            \propto
            \exp\left[
            -\frac{1}{2}
            \left(
            \frac{(x-\mu_1)^2}{\sigma_1^2}
            +
            \frac{(x-\mu_2)^2}{\sigma_2^2}
            \right)
            \right].
          \]</div>
          <p>把括号展开，只看 \(x\) 的二次项和一次项：</p>
          <div class="equation">\[
            \frac{(x-\mu_1)^2}{\sigma_1^2}
            +
            \frac{(x-\mu_2)^2}{\sigma_2^2}
            =
            \left(\frac{1}{\sigma_1^2}+\frac{1}{\sigma_2^2}\right)x^2
            -
            2\left(\frac{\mu_1}{\sigma_1^2}+\frac{\mu_2}{\sigma_2^2}\right)x
            +
            \mathrm{const}.
          \]</div>
          <p>令精度，也就是方差的倒数，为 \(\lambda_i=1/\sigma_i^2\)。新高斯的精度是精度相加：</p>
          <div class="equation">\[
            \lambda=\lambda_1+\lambda_2,
            \qquad
            \sigma^2=\frac{1}{\lambda_1+\lambda_2}.
          \]</div>
          <p>新均值是按精度加权的平均：</p>
          <div class="equation">\[
            \mu=\frac{\lambda_1\mu_1+\lambda_2\mu_2}{\lambda_1+\lambda_2}.
          \]</div>
          <p>这条事实后面会直接用来推导 \(q(x_{t-1}|x_t,x_0)\)。你可以把它理解成两个信息源在投票：方差小的信息源更可靠，因此权重更大。</p>
        </section>

        <section class="article-section">
          <h2>3. DDPM 的前向加噪过程</h2>
          <p>DDPM 把前向过程定义成一个 Markov 链：</p>
          <div class="equation">\[
            x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_T.
          \]</div>
          <p>其中 \(x_0\) 是真实数据，\(x_T\) 希望接近标准高斯。Markov 性的意思是：给定 \(x_{t-1}\) 后，\(x_t\) 不再依赖更早的状态：</p>
          <div class="equation">\[
            q(x_t|x_{t-1},x_{t-2},\ldots,x_0)=q(x_t|x_{t-1}).
          \]</div>
          <p>DDPM 选择每一步加入一点高斯噪声：</p>
          <div class="equation">\[
            q(x_t|x_{t-1})=\mathcal{N}\left(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I\right),
            \qquad \alpha_t=1-\beta_t.
          \]</div>
          <p>这等价于下面的采样形式：</p>
          <div class="equation">\[
            x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t,\qquad \epsilon_t\sim\mathcal{N}(0,I).
          \]</div>
          <p>为什么要乘 \(\sqrt{\alpha_t}\)，而不是简单写成 \(x_t=x_{t-1}+\mathrm{noise}\)？原因是尺度控制。假设 \(x_{t-1}\) 的方差大约是 1，那么 \(\sqrt{\alpha_t}x_{t-1}\) 的方差约为 \(\alpha_t\)，噪声项 \(\sqrt{\beta_t}\epsilon_t\) 的方差约为 \(\beta_t\)。二者相加的方差约为 \(\alpha_t+\beta_t=1\)。这让数据在加噪过程中不会无限膨胀。</p>

          <figure class="article-figure chain-figure">
            <div class="chain-node clean">\(x_0\)<span>真实样本</span></div>
            <div class="chain-arrow">→</div>
            <div class="chain-node">\(x_1\)<span>轻微噪声</span></div>
            <div class="chain-arrow">→</div>
            <div class="chain-node">\(x_t\)<span>中等噪声</span></div>
            <div class="chain-arrow">→</div>
            <div class="chain-node noise">\(x_T\)<span>近似高斯</span></div>
            <figcaption>前向过程逐步削弱数据结构，同时增加随机性。训练时我们完全知道这条加噪路径。</figcaption>
          </figure>
        </section>

        <section class="article-section">
          <h2>4. 关键闭式公式：直接从 \(x_0\) 得到 \(x_t\)</h2>
          <p>如果每次训练都从 \(x_0\) 一步一步采样到 \(x_t\)，会很麻烦。幸运的是，DDPM 的前向过程有一个闭式公式：</p>
          <div class="equation">\[
            q(x_t|x_0)=\mathcal{N}\left(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\right),
            \qquad \bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s.
          \]</div>
          <p>等价采样形式是：</p>
          <div class="equation">\[
            x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I).
          \]</div>
          <p>下面完整推导。我们先看两步。</p>
          <div class="equation">\[
            x_1=\sqrt{\alpha_1}x_0+\sqrt{\beta_1}\epsilon_1.
          \]</div>
          <div class="equation">\[
            x_2=\sqrt{\alpha_2}x_1+\sqrt{\beta_2}\epsilon_2.
          \]</div>
          <p>把第一式代入第二式：</p>
          <div class="equation">\[
            x_2=\sqrt{\alpha_2\alpha_1}x_0+\sqrt{\alpha_2\beta_1}\epsilon_1+\sqrt{\beta_2}\epsilon_2.
          \]</div>
          <p>两个独立标准高斯的线性组合仍然是高斯。噪声总方差等于系数平方和：</p>
          <div class="equation">\[
            \alpha_2\beta_1+\beta_2=\alpha_2(1-\alpha_1)+(1-\alpha_2)=1-\alpha_1\alpha_2.
          \]</div>
          <p>于是：</p>
          <div class="equation">\[
            x_2=\sqrt{\alpha_1\alpha_2}x_0+\sqrt{1-\alpha_1\alpha_2}\epsilon.
          \]</div>
          <p>一般 \(t\) 步可以用归纳法。假设：</p>
          <div class="equation">\[
            x_{t-1}=\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}\epsilon.
          \]</div>
          <p>代入 \(x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t\)。信号项变为：</p>
          <div class="equation">\[
            \sqrt{\alpha_t}\sqrt{\bar{\alpha}_{t-1}}x_0=\sqrt{\bar{\alpha}_t}x_0.
          \]</div>
          <p>噪声方差变为：</p>
          <div class="equation">\[
            \alpha_t(1-\bar{\alpha}_{t-1})+\beta_t
            =\alpha_t-\bar{\alpha}_t+1-\alpha_t
            =1-\bar{\alpha}_t.
          \]</div>
          <p>所以闭式公式成立。</p>
          <div class="insight-box">这个公式是 DDPM 训练能高效进行的关键。训练时我们可以随机抽一个 \(t\)，一次性合成 \(x_t\)，不需要真的执行 \(t\) 次加噪。</div>

          <figure class="source-figure">
            <img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-beta.png" alt="线性和余弦噪声日程对比" />
            <figcaption>参考图：线性 schedule 与 cosine schedule 下 \(\bar{\alpha}_t\) 的衰减。它帮助理解不同时间步保留多少原始信号。图片来源：Lilian Weng, What are Diffusion Models?</figcaption>
          </figure>

          <div class="viz-card noise-viz" data-viz="noise">
            <canvas width="720" height="420" aria-label="前向加噪二维可视化"></canvas>
            <div class="viz-controls">
              <label>时间步 t <input type="range" min="0" max="100" value="0" /></label>
              <output>t = 0, 信号强</output>
            </div>
          </div>
        </section>

        <section class="article-section">
          <h2>4.5 从信噪比理解时间步</h2>
          <p>闭式公式也可以写成：</p>
          <div class="equation">\[
            x_t=\underbrace{\sqrt{\bar{\alpha}_t}x_0}_{\text{信号}}
            +
            \underbrace{\sqrt{1-\bar{\alpha}_t}\epsilon}_{\text{噪声}}.
          \]</div>
          <p>所以第 \(t\) 步不是一个抽象编号，而是一个噪声等级。衡量噪声等级最自然的量是信噪比：</p>
          <div class="equation">\[
            \mathrm{SNR}(t)=
            \frac{\text{信号方差}}{\text{噪声方差}}
            =
            \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}.
          \]</div>
          <p>当 \(t\) 很小，\(\bar{\alpha}_t\) 接近 1，SNR 很大，说明样本里主要是原图信息；当 \(t\) 很大，\(\bar{\alpha}_t\) 接近 0，SNR 很小，说明样本几乎全是噪声。很多论文讨论 schedule，本质上就是讨论模型应该在不同 SNR 区间看到多少训练样本，以及采样时如何分配步长。</p>
          <p>线性 \(\beta_t\) schedule 很直观，但不一定让每个噪声等级都被合理利用。cosine schedule 的动机之一，是让 \(\bar{\alpha}_t\) 的衰减更平滑，避免前期或后期变化过快。读 improved DDPM、EDM、flow matching 或 consistency model 时，经常会看到作者重新设计噪声参数化；这些设计通常都可以回到 SNR 来理解。</p>
          <div class="insight-box">看到新的噪声 schedule 时，不要先背公式。先问三个问题：低噪声区有多细？高噪声区有多细？中间 SNR 区间是否给了模型足够训练信号？</div>
        </section>

        <section class="article-section">
          <h2>5. 反向过程：我们真正要学习的东西</h2>
          <p>生成时我们没有 \(x_0\)。我们只有一个随机噪声 \(x_T\sim\mathcal{N}(0,I)\)，希望一步步得到 \(x_{T-1},x_{T-2},\ldots,x_0\)。因此模型定义为：</p>
          <div class="equation">\[
            p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_t).
          \]</div>
          <p>真正困难的是 \(q(x_{t-1}|x_t)\)。只看一个带噪样本 \(x_t\)，并不知道它来自哪个干净样本；许多不同的 \(x_0\) 都可能加噪到相似的 \(x_t\)。</p>
          <p>训练时我们知道 \(x_0\)，所以先研究更容易的后验：</p>
          <div class="equation">\[
            q(x_{t-1}|x_t,x_0).
          \]</div>
          <p>这个分布可以精确推导成高斯。然后我们让神经网络学习一个高斯反向转移来近似它。</p>

          <figure class="source-figure">
            <img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png" alt="DDPM 前向加噪和反向去噪示意图" />
            <figcaption>参考图：DDPM 的前向扩散和反向去噪链。图片来源：Lilian Weng, What are Diffusion Models?</figcaption>
          </figure>
        </section>

        <section class="article-section">
          <h2>6. 后验 \(q(x_{t-1}|x_t,x_0)\) 的推导</h2>
          <p>根据贝叶斯公式：</p>
          <div class="equation">\[
            q(x_{t-1}|x_t,x_0)=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}.
          \]</div>
          <p>由于前向过程是 Markov 的，\(q(x_t|x_{t-1},x_0)=q(x_t|x_{t-1})\)。分母 \(q(x_t|x_0)\) 与未知变量 \(x_{t-1}\) 无关，所以如果只关心关于 \(x_{t-1}\) 的分布形状，可以写成：</p>
          <div class="equation">\[
            q(x_{t-1}|x_t,x_0)\propto q(x_t|x_{t-1})q(x_{t-1}|x_0).
          \]</div>
          <p>两个因子分别是：</p>
          <div class="equation">\[
            q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I),
          \]</div>
          <div class="equation">\[
            q(x_{t-1}|x_0)=\mathcal{N}(x_{t-1};\sqrt{\bar{\alpha}_{t-1}}x_0,(1-\bar{\alpha}_{t-1})I).
          \]</div>
          <p>令 \(x=x_{t-1}\)。高斯密度的指数部分只看和 \(x\) 有关的项：</p>
          <div class="equation">\[
            -\frac{1}{2\beta_t}\|x_t-\sqrt{\alpha_t}x\|^2
            -\frac{1}{2(1-\bar{\alpha}_{t-1})}\|x-\sqrt{\bar{\alpha}_{t-1}}x_0\|^2.
          \]</div>
          <p>展开第一项：</p>
          <div class="equation">\[
            \|x_t-\sqrt{\alpha_t}x\|^2
            =x_t^\top x_t-2\sqrt{\alpha_t}x_t^\top x+\alpha_t x^\top x.
          \]</div>
          <p>展开第二项：</p>
          <div class="equation">\[
            \|x-\sqrt{\bar{\alpha}_{t-1}}x_0\|^2
            =x^\top x-2\sqrt{\bar{\alpha}_{t-1}}x_0^\top x+\bar{\alpha}_{t-1}x_0^\top x_0.
          \]</div>
          <p>去掉和 \(x\) 无关的常数，得到：</p>
          <div class="equation">\[
            -\frac{1}{2}\left[
            \left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}}\right)x^\top x
            -2\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t+\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0\right)^\top x
            \right].
          \]</div>
          <p>这就是一个高斯分布的标准二次形式。为了看清楚，我们把它写成：</p>
          <div class="equation">\[
            -\frac{1}{2}\left[A x^\top x-2b^\top x\right],
          \]</div>
          <p>其中：</p>
          <div class="equation">\[
            A=
            \frac{\alpha_t}{\beta_t}
            +
            \frac{1}{1-\bar{\alpha}_{t-1}},
            \qquad
            b=
            \frac{\sqrt{\alpha_t}}{\beta_t}x_t
            +
            \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0.
          \]</div>
          <p>对一个各向同性高斯来说，二次项前的 \(A\) 是精度，也就是方差的倒数。因此后验方差应该是 \(A^{-1}\)。现在把 \(A\) 化简：</p>
          <div class="equation">\[
            A
            =
            \frac{\alpha_t(1-\bar{\alpha}_{t-1})+\beta_t}
            {\beta_t(1-\bar{\alpha}_{t-1})}.
          \]</div>
          <p>分子继续化简：</p>
          <div class="equation">\[
            \alpha_t(1-\bar{\alpha}_{t-1})+\beta_t
            =
            \alpha_t-\alpha_t\bar{\alpha}_{t-1}+1-\alpha_t
            =
            1-\bar{\alpha}_t.
          \]</div>
          <p>所以：</p>
          <div class="equation">\[
            A=
            \frac{1-\bar{\alpha}_t}
            {\beta_t(1-\bar{\alpha}_{t-1})},
            \qquad
            A^{-1}=
            \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}.
          \]</div>
          <p>这就得到后验方差。后验均值来自配方。因为：</p>
          <div class="equation">\[
            A x^\top x-2b^\top x
            =
            A\left\|x-\frac{b}{A}\right\|^2+\mathrm{const},
          \]</div>
          <p>所以均值是 \(A^{-1}b\)。把 \(A^{-1}\) 乘到 \(b\) 上，就会得到两个权重：一个乘 \(x_0\)，一个乘 \(x_t\)。整理后：</p>
          <div class="equation">\[
            q(x_{t-1}|x_t,x_0)=\mathcal{N}\left(x_{t-1};\tilde{\mu}_t(x_t,x_0),\tilde{\beta}_t I\right),
          \]</div>
          <div class="equation">\[
            \tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t,
          \]</div>
          <div class="equation">\[
            \tilde{\mu}_t(x_t,x_0)
            =
            \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0
            +
            \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t.
          \]</div>
          <p>这个结果值得停下来理解：最优后验均值是 \(x_0\) 和 \(x_t\) 的加权组合。低噪声时 \(x_t\) 已经很接近 \(x_0\)，高噪声时需要更多依赖对干净样本的估计。</p>
        </section>

        <section class="article-section">
          <h2>7. 为什么 DDPM 通常让网络预测噪声？</h2>
          <p>我们已经知道：</p>
          <div class="equation">\[
            x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon.
          \]</div>
          <p>如果网络能预测加入的噪声 \(\epsilon\)，就可以反解出对 \(x_0\) 的估计：</p>
          <div class="equation">\[
            \hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}.
          \]</div>
          <p>再把 \(\hat{x}_0\) 代入 \(\tilde{\mu}_t(x_t,x_0)\)，就得到反向采样的均值。</p>
          <p>训练时，真实噪声 \(\epsilon\) 是我们自己采样出来的，所以可以直接监督：</p>
          <div class="equation">\[
            L_{\mathrm{simple}}(\theta)=
            \mathbb{E}_{x_0,t,\epsilon}
            \left[
            \|\epsilon-\epsilon_\theta(x_t,t)\|^2
            \right].
          \]</div>
          <p>这就是 DDPM 中最常用的简化训练目标。它表面上是一个普通 MSE，但背后对应的是变分下界中每个反向高斯转移的 KL 项。DDPM 论文发现，去掉复杂权重后用这个简单目标训练，图像质量反而很好。</p>
          <div class="insight-box">预测噪声并不是唯一选择。模型也可以预测 \(x_0\)、score、或者 velocity \(v\)。这些参数化之间可以互相转换；不同选择会影响训练稳定性、采样器设计和不同噪声等级下的误差放大方式。</div>
        </section>

        <section class="article-section">
          <h2>7.5 训练算法：把推导翻译成代码会发生什么？</h2>
          <p>理论推导到这里，训练过程其实已经很简单。每一次迭代做四件事：</p>
          <ol>
            <li>从训练集中取一批真实样本 \(x_0\)。</li>
            <li>随机采样时间步 \(t\)，通常从 \(\{1,\ldots,T\}\) 均匀采样。</li>
            <li>采样标准高斯噪声 \(\epsilon\)，用闭式公式合成 \(x_t\)。</li>
            <li>让网络 \(\epsilon_\theta(x_t,t)\) 预测这份噪声，用 MSE 更新参数。</li>
          </ol>
          <p>写成数学形式就是：</p>
          <div class="equation">\[
            t\sim \mathrm{Uniform}\{1,\ldots,T\},
            \qquad
            \epsilon\sim\mathcal{N}(0,I),
          \]</div>
          <div class="equation">\[
            x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,
          \]</div>
          <div class="equation">\[
            \theta \leftarrow
            \theta-\eta\nabla_\theta
            \left\|
            \epsilon-\epsilon_\theta(x_t,t)
            \right\|^2.
          \]</div>
          <p>这个过程看起来像“训练一个去噪网络”，但它比普通图像去噪更强：普通去噪通常固定某个噪声等级，而 Diffusion 会在很多噪声等级上训练同一个网络。因此模型学到的是一族去噪问题，而不是单个去噪任务。</p>

          <h3>\(\epsilon\)-prediction、\(x_0\)-prediction 和 \(v\)-prediction</h3>
          <p>设：</p>
          <div class="equation">\[
            \sigma_t=\sqrt{1-\bar{\alpha}_t},
            \qquad
            a_t=\sqrt{\bar{\alpha}_t}.
          \]</div>
          <p>闭式加噪写成 \(x_t=a_tx_0+\sigma_t\epsilon\)。如果模型预测噪声，就可以得到：</p>
          <div class="equation">\[
            \hat{x}_0=\frac{x_t-\sigma_t\epsilon_\theta(x_t,t)}{a_t}.
          \]</div>
          <p>如果模型直接预测 \(x_0\)，则可以反推出噪声：</p>
          <div class="equation">\[
            \hat{\epsilon}=\frac{x_t-a_t x_{\theta,0}(x_t,t)}{\sigma_t}.
          \]</div>
          <p>很多现代实现还使用 \(v\)-prediction。一个常见定义是：</p>
          <div class="equation">\[
            v=a_t\epsilon-\sigma_t x_0.
          \]</div>
          <p>它把噪声和干净样本混合起来。为什么要这样做？因为在低噪声区，直接预测 \(\epsilon\) 可能让很小的噪声误差被放大；在高噪声区，直接预测 \(x_0\) 又非常困难。\(v\) 参数化在不同噪声等级之间更均衡一些。</p>
          <p>从 \(x_t\) 和 \(v\) 可以恢复 \(x_0\) 与 \(\epsilon\)：</p>
          <div class="equation">\[
            x_0=a_t x_t-\sigma_t v,
            \qquad
            \epsilon=\sigma_t x_t+a_t v.
          \]</div>
          <p>这里不要死背。只要记住 \(x_t=a_tx_0+\sigma_t\epsilon\) 和 \(v=a_t\epsilon-\sigma_tx_0\)，它们组成一个二维旋转式的线性变换，代数上可以互相解出。</p>
        </section>

        <section class="article-section">
          <h2>8. 从噪声预测到 score matching</h2>
          <p>score 定义为概率密度对输入的 log 梯度：</p>
          <div class="equation">\[
            s_t(x)=\nabla_x\log p_t(x).
          \]</div>
          <p>它的直觉是：告诉我们当前位置往哪个方向移动，概率密度上升最快。对高斯分布来说，score 会把点拉回均值附近。</p>
          <p>考虑条件分布：</p>
          <div class="equation">\[
            q(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I).
          \]</div>
          <p>对 \(x_t\) 求梯度：</p>
          <div class="equation">\[
            \nabla_{x_t}\log q(x_t|x_0)
            =
            -\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}.
          \]</div>
          <p>又因为：</p>
          <div class="equation">\[
            x_t-\sqrt{\bar{\alpha}_t}x_0=\sqrt{1-\bar{\alpha}_t}\epsilon,
          \]</div>
          <p>所以：</p>
          <div class="equation">\[
            \nabla_{x_t}\log q(x_t|x_0)
            =
            -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
          \]</div>
          <p>这说明预测噪声 \(\epsilon\) 与预测条件 score 只差一个已知的尺度因子。DDPM 和 score-based model 的核心联系就在这里：二者都在学习 noisy distribution 的方向信息。</p>
          <p>不过论文里真正需要的是边缘分布 \(p_t(x_t)\) 的 score，而不是给定 \(x_0\) 后的条件 score。边缘分布是：</p>
          <div class="equation">\[
            p_t(x_t)=\int q(x_t|x_0)p_{\mathrm{data}}(x_0)dx_0.
          \]</div>
          <p>这个积分表示：所有可能的真实样本 \(x_0\) 都可能加噪得到当前 \(x_t\)。直接求 \(\nabla_{x_t}\log p_t(x_t)\) 很难，因为真实数据分布未知。但 denoising score matching 给了一个可训练的办法：用条件 score 作为监督，学习一个网络 \(s_\theta(x_t,t)\)，使它在期望意义下接近边缘 score。</p>
          <div class="equation">\[
            \mathbb{E}_{p_{\mathrm{data}}(x_0)q(x_t|x_0)}
            \left[
            \left\|
            s_\theta(x_t,t)
            -
            \nabla_{x_t}\log q(x_t|x_0)
            \right\|^2
            \right].
          \]</div>
          <p>直觉是：同一个 \(x_t\) 可能由很多 \(x_0\) 产生，训练时模型看见大量这样的配对后，会学到这些条件方向的平均效果。这个平均方向正是把样本推回高概率区域的边缘 score。</p>
          <div class="insight-box">所以“预测噪声”不是一个随便挑的工程技巧。它是用一个容易构造的监督信号，间接学习每个噪声等级上的 score。</div>

          <figure class="article-figure score-figure">
            <svg viewBox="0 0 720 260" role="img" aria-label="score 场示意图">
              <defs>
                <marker id="score-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L9,3 z" fill="currentColor"></path>
                </marker>
              </defs>
              <g class="score-cluster left"><circle cx="230" cy="130" r="58"></circle><circle cx="230" cy="130" r="24"></circle></g>
              <g class="score-cluster right"><circle cx="500" cy="130" r="58"></circle><circle cx="500" cy="130" r="24"></circle></g>
              <g class="score-arrows">
                <line x1="90" y1="60" x2="175" y2="105"></line>
                <line x1="110" y1="205" x2="180" y2="155"></line>
                <line x1="350" y1="58" x2="285" y2="108"></line>
                <line x1="360" y1="210" x2="285" y2="158"></line>
                <line x1="630" y1="60" x2="555" y2="105"></line>
                <line x1="620" y1="205" x2="555" y2="155"></line>
              </g>
            </svg>
            <figcaption>score 指向高密度区域。对生成而言，它提供了“从噪声往数据走”的局部方向。</figcaption>
          </figure>
        </section>

        <section class="article-section">
          <h2>9. 变分下界：DDPM 训练目标从哪里来？</h2>
          <p>如果我们希望最大化模型对数据的似然 \(p_\theta(x_0)\)，直接计算通常很困难，因为模型包含整条隐变量链 \(x_{1:T}\)。于是使用变分推断。我们从边缘似然开始：</p>
          <div class="equation">\[
            p_\theta(x_0)=\int p_\theta(x_{0:T})dx_{1:T}.
          \]</div>
          <p>引入前向过程 \(q(x_{1:T}|x_0)\) 作为变分分布：</p>
          <div class="equation">\[
            \log p_\theta(x_0)
            =
            \log \int q(x_{1:T}|x_0)\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}dx_{1:T}.
          \]</div>
          <p>把积分写成期望：</p>
          <div class="equation">\[
            \log p_\theta(x_0)
            =
            \log \mathbb{E}_{q}
            \left[
            \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}
            \right].
          \]</div>
          <p>由于 \(\log\) 是凹函数，Jensen 不等式给出：</p>
          <div class="equation">\[
            \log \mathbb{E}_q[Y]\geq \mathbb{E}_q[\log Y].
          \]</div>
          <p>所以：</p>
          <div class="equation">\[
            \log p_\theta(x_0)
            \geq
            \mathbb{E}_q
            \left[
            \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}
            \right].
          \]</div>
          <p>这就是 ELBO。DDPM 通常最小化负的 ELBO，因此常写成变分上界 \(L_{\mathrm{vlb}}\)。把前向链和反向链展开，可以分解成：</p>
          <div class="equation">\[
            L_{\mathrm{vlb}}
            =
            L_T+\sum_{t=2}^{T}L_{t-1}+L_0.
          \]</div>
          <p>这三个部分分别有不同含义。</p>
          <div class="equation">\[
            L_T=
            D_{\mathrm{KL}}\left(q(x_T|x_0)\,\|\,p(x_T)\right).
          \]</div>
          <p>\(L_T\) 要求最后一步的前向分布接近标准高斯。如果 \(T\) 足够大且 schedule 合理，\(q(x_T|x_0)\) 已经非常接近 \(\mathcal{N}(0,I)\)，这一项通常没有可学习参数。</p>
          <div class="equation">\[
            L_{t-1}=
            D_{\mathrm{KL}}
            \left(
            q(x_{t-1}|x_t,x_0)
            \,\|\,
            p_\theta(x_{t-1}|x_t)
            \right),
            \qquad t=2,\ldots,T.
          \]</div>
          <p>这是最核心的项。它要求模型反向一步 \(p_\theta(x_{t-1}|x_t)\) 接近真实后验 \(q(x_{t-1}|x_t,x_0)\)。前面我们推导真实后验，就是为了知道这个 KL 项应该匹配什么。</p>
          <div class="equation">\[
            L_0=-\log p_\theta(x_0|x_1).
          \]</div>
          <p>\(L_0\) 负责最后从轻微噪声 \(x_1\) 重建 \(x_0\)。在图像模型里，它和离散像素似然或重建误差有关。</p>
          <p>如果 \(p_\theta(x_{t-1}|x_t)\) 和 \(q(x_{t-1}|x_t,x_0)\) 都是高斯，并且方差固定，那么两个高斯的 KL 主要变成均值之间的平方误差：</p>
          <div class="equation">\[
            L_{t-1}
            =
            \mathbb{E}_q
            \left[
            \frac{1}{2\sigma_t^2}
            \left\|
            \tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t,t)
            \right\|^2
            \right]
            +\mathrm{const}.
          \]</div>
          <p>这里的常数不依赖 \(\theta\)，训练时可以忽略。接下来用噪声预测参数化 \(\mu_\theta\)。因为 \(\hat{x}_0\) 可以由 \(\epsilon_\theta\) 反解出来，\(\mu_\theta\) 也就可以由 \(\epsilon_\theta\) 表示。代入后，均值误差会变成噪声误差的加权形式：</p>
          <div class="equation">\[
            L_{t-1}
            =
            \mathbb{E}_{x_0,\epsilon}
            \left[
            w_t
            \left\|
            \epsilon-\epsilon_\theta(x_t,t)
            \right\|^2
            \right]
            +\mathrm{const}.
          \]</div>
          <p>\(w_t\) 是只依赖 schedule 的权重。DDPM 论文中的 \(L_{\mathrm{simple}}\) 进一步去掉这个权重，直接训练：</p>
          <div class="equation">\[
            L_{\mathrm{simple}}=
            \mathbb{E}_{x_0,t,\epsilon}
            \left[
            \left\|
            \epsilon-\epsilon_\theta(x_t,t)
            \right\|^2
            \right].
          \]</div>
          <p>初学者读论文时最容易卡在这里：为什么突然从 likelihood 变成预测噪声？中间桥梁就是：likelihood 下界 → 反向高斯 KL → 均值匹配 → 用 \(\epsilon_\theta\) 参数化均值 → MSE。</p>
          <div class="paper-note">如果你想读 DDPM 原论文，可以先把所有 \(L_T,L_{t-1},L_0\) 都标出来，再逐个问：这一项有没有参数？它要求哪个分布接近哪个分布？它最后会不会变成一个可训练的 MSE？这样论文里的推导会清楚很多。</div>
        </section>

        <section class="article-section">
          <h2>10. 采样：DDPM 与 DDIM</h2>
          <p>训练完成后，DDPM 采样从 \(x_T\sim\mathcal{N}(0,I)\) 开始。对 \(t=T,T-1,\ldots,1\)：</p>
          <ol>
            <li>用网络预测 \(\epsilon_\theta(x_t,t)\)。</li>
            <li>由预测噪声得到 \(\hat{x}_0\)。</li>
            <li>计算反向均值 \(\mu_\theta(x_t,t)\)。</li>
            <li>加入适当方差的高斯噪声，得到 \(x_{t-1}\)。</li>
          </ol>
          <p>DDPM 的反向过程是随机的 Markov 链。DDIM 则构造了一个可以确定性采样的路径。它保持相同的边缘 \(q(x_t|x_0)\)，但不要求反向过程必须是原来的随机 Markov 链。当 \(\eta=0\) 时，DDIM 采样是确定性的；同一个初始噪声会得到一致的生成结果。这也解释了为什么 DDIM 可以用于 latent interpolation。</p>
          <p>DDIM 的一步更新常写成下面的形式。先由网络得到：</p>
          <div class="equation">\[
            \hat{x}_0=
            \frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}
            {\sqrt{\bar{\alpha}_t}}.
          \]</div>
          <p>然后从 \(t\) 走到较小的时间 \(s\)：</p>
          <div class="equation">\[
            x_s=
            \sqrt{\bar{\alpha}_s}\hat{x}_0
            +
            \sqrt{1-\bar{\alpha}_s-\sigma_{t\rightarrow s}^2}\,
            \epsilon_\theta(x_t,t)
            +
            \sigma_{t\rightarrow s}z,
            \qquad z\sim\mathcal{N}(0,I).
          \]</div>
          <p>当 \(\sigma_{t\rightarrow s}=0\) 时，这一步没有额外随机噪声，就是确定性更新；当 \(\sigma_{t\rightarrow s}\) 取合适值时，又可以恢复类似 DDPM 的随机性。DDIM 的重要意义是：训练时仍然用 DDPM 的噪声预测目标，但采样时可以跳步，用更少步数完成生成。这打开了后续各种快速采样器的大门。</p>
          <p>从学习角度看，DDIM 还揭示了一件事：模型真正学到的不是某条唯一的随机链，而是每个噪声等级上的方向信息。只要新的采样路径使用这些方向信息，并且边缘分布设计得合理，就可能生成好样本。</p>
        </section>

        <section class="article-section">
          <h2>11. SDE 与 ODE：连续时间视角</h2>
          <p>当时间步足够细时，离散扩散过程可以写成随机微分方程：</p>
          <div class="equation">\[
            dx=f(x,t)dt+g(t)dw.
          \]</div>
          <p>这里 \(f(x,t)\) 是 drift，表示确定性趋势；\(g(t)dw\) 是随机噪声项。前向 SDE 把数据变成噪声。反向 SDE 需要 score：</p>
          <div class="equation">\[
            dx=\left[f(x,t)-g(t)^2\nabla_x\log p_t(x)\right]dt+g(t)d\bar{w}.
          \]</div>
          <p>这条公式的核心含义是：如果知道每个时间 \(t\) 的 score，就能从噪声分布反向走回数据分布。</p>
          <p>同一个边缘分布族还对应一个 probability flow ODE：</p>
          <div class="equation">\[
            dx=\left[f(x,t)-\frac{1}{2}g(t)^2\nabla_x\log p_t(x)\right]dt.
          \]</div>
          <p>它没有随机项，但在每个时间点拥有和 SDE 相同的边缘分布 \(p_t(x)\)。这解释了为什么可以用确定性 ODE 采样器生成样本，也解释了 DDIM 与 probability flow 之间的联系。</p>

          <h3>VP SDE：DDPM 的连续版本</h3>
          <p>DDPM 常对应 variance preserving SDE，也就是 VP SDE：</p>
          <div class="equation">\[
            dx=-\frac{1}{2}\beta(t)x\,dt+\sqrt{\beta(t)}\,dw.
          \]</div>
          <p>它叫 variance preserving，是因为在合适条件下，过程不会让总体尺度无限变大。这个式子和离散 DDPM 的 \(x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t\) 是同一种思想的连续极限：第一项让信号逐渐衰减，第二项加入噪声。</p>
          <p>VP SDE 的边缘分布仍有类似闭式：</p>
          <div class="equation">\[
            x(t)=\alpha(t)x(0)+\sigma(t)\epsilon,
          \]</div>
          <p>其中 \(\alpha(t)\) 随时间下降，\(\sigma(t)\) 随时间上升。离散 DDPM 的 \(\sqrt{\bar{\alpha}_t}\) 和 \(\sqrt{1-\bar{\alpha}_t}\)，就是这个连续形式的离散对应。</p>

          <h3>为什么反向 SDE 里会出现 score？</h3>
          <p>直觉上，前向 SDE 不断加入噪声，使分布越来越平滑。反向时，如果只把时间倒过来，会缺少“往数据高密度区域聚拢”的信息。score 正好提供这个信息：它告诉当前 \(x\) 在分布 \(p_t\) 下往哪里移动会让 log density 增大。</p>
          <p>因此反向漂移项不是简单的 \(-f(x,t)\)，而要额外减去 \(g(t)^2\nabla_x\log p_t(x)\)。这个项会抵消扩散造成的概率流失，把样本推回数据结构更强的位置。</p>

          <h3>连续性方程：ODE 和 Flow Matching 的公共语言</h3>
          <p>如果粒子按照速度场 \(v_t(x)\) 运动：</p>
          <div class="equation">\[
            \frac{dx}{dt}=v_t(x),
          \]</div>
          <p>那么它们形成的概率密度 \(p_t(x)\) 会满足连续性方程：</p>
          <div class="equation">\[
            \frac{\partial p_t(x)}{\partial t}
            =
            -\nabla\cdot\left(p_t(x)v_t(x)\right).
          \]</div>
          <p>这个式子的意思很朴素：一个区域里的概率质量会因为速度场流入而增加，因为速度场流出而减少。probability flow ODE 和 flow matching 都可以用这句话理解，只是它们得到速度场的方式不同。</p>
        </section>

        <section class="article-section">
          <h2>12. Flow Matching：直接学习速度场</h2>
          <p>Flow Matching 从另一个角度看生成：我们不一定先学习 score，再由 score 构造反向 SDE/ODE；也可以直接学习一个速度场：</p>
          <div class="equation">\[
            \frac{dx_t}{dt}=v_\theta(x_t,t).
          \]</div>
          <p>最简单的路径是把噪声样本 \(x_0\) 和数据样本 \(x_1\) 线性插值：</p>
          <div class="equation">\[
            x_t=(1-t)x_0+tx_1.
          \]</div>
          <p>这条路径的速度是：</p>
          <div class="equation">\[
            u_t=\frac{dx_t}{dt}=x_1-x_0.
          \]</div>
          <p>训练时让网络回归这个速度：</p>
          <div class="equation">\[
            L(\theta)=\mathbb{E}\left[\|v_\theta(x_t,t)-u_t\|^2\right].
          \]</div>
          <p>更一般的 Flow Matching 使用不同 probability path 和条件速度场。这里要区分两个概念。</p>
          <p><strong>条件路径</strong> \(p_t(x|z)\)：给定某个条件变量 \(z\)，比如一对噪声和数据样本，定义一条从源分布到目标分布的路径。</p>
          <p><strong>边缘路径</strong> \(p_t(x)\)：把所有条件变量积分掉后，真实样本总体在时间 \(t\) 的分布。</p>
          <div class="equation">\[
            p_t(x)=\int p_t(x|z)p(z)dz.
          \]</div>
          <p>训练时我们通常知道条件速度 \(u_t(x|z)\)，但生成时只能看到 \(x_t\)，不知道它对应哪个 \(z\)。Flow Matching 的关键结果是：直接回归条件速度，在期望意义下可以学到边缘速度场：</p>
          <div class="equation">\[
            v_t(x)=\mathbb{E}\left[u_t(x|z)\mid x_t=x\right].
          \]</div>
          <p>这和 denoising score matching 的味道很像：训练监督来自条件对象，但模型最终学到的是边缘对象。Diffusion 里是条件 score 到边缘 score；Flow Matching 里是条件速度到边缘速度。</p>
          <p>DiffusionFlow 的文章强调：在高斯源分布和合适 schedule 下，diffusion 与 flow matching 可以相互转换。它们常常不是“谁替代谁”的关系，而是同一类分布运输思想的不同参数化。</p>

          <h3>Flow Matching 和 Diffusion 的区别在哪里？</h3>
          <p>Diffusion 的经典训练目标经常从“加噪和去噪”出发，学习 score 或噪声；Flow Matching 则从“概率质量如何被速度场运输”出发，直接学习 ODE 速度。二者都可以从简单分布生成复杂分布，但思考入口不同。</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>问题</th><th>Diffusion/Score 视角</th><th>Flow Matching 视角</th></tr>
              </thead>
              <tbody>
                <tr><td>路径怎么来？</td><td>先设计加噪过程，让数据逐渐变成高斯。</td><td>直接设计从源分布到数据分布的 probability path。</td></tr>
                <tr><td>网络学什么？</td><td>噪声、score、\(x_0\) 或 \(v\) 参数化。</td><td>速度场 \(v_t(x)\)。</td></tr>
                <tr><td>生成怎么做？</td><td>反向 SDE、probability flow ODE 或各种离散采样器。</td><td>求解 ODE \(\frac{dx}{dt}=v_\theta(x,t)\)。</td></tr>
                <tr><td>核心方程</td><td>反向 SDE / score identity。</td><td>连续性方程。</td></tr>
              </tbody>
            </table>
          </div>

          <figure class="article-figure flow-figure">
            <svg viewBox="0 0 720 280" role="img" aria-label="Flow Matching 轨迹示意图">
              <defs>
                <marker id="flow-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L9,3 z" fill="#16b4a5"></path>
                </marker>
              </defs>
              <path class="flow-path" d="M100 210 C190 70, 310 230, 430 95 S610 70, 650 165"></path>
              <circle class="flow-dot noise-dot" cx="100" cy="210" r="12"></circle>
              <circle class="flow-dot data-dot" cx="650" cy="165" r="12"></circle>
              <text x="70" y="245">噪声</text>
              <text x="622" y="205">数据</text>
              <line class="flow-vector" x1="315" y1="154" x2="386" y2="117"></line>
              <text x="360" y="96">速度场 v(x,t)</text>
            </svg>
            <figcaption>Flow Matching 把生成看成连续运输：从简单噪声分布出发，沿速度场流向数据分布。</figcaption>
          </figure>
        </section>

        <section class="article-section">
          <h2>13. 几种模型输出的统一理解</h2>
          <p>论文里常见不同输出：预测噪声 \(\epsilon\)、预测干净样本 \(x_0\)、预测 score、预测 velocity \(v\)、预测 flow \(u\)。它们不是互不相干的概念，而是同一个 noisy sample 的不同坐标表达。</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>输出</th><th>含义</th><th>优点</th><th>常见位置</th></tr>
              </thead>
              <tbody>
                <tr><td>\(\epsilon\)</td><td>加入到 \(x_0\) 的噪声</td><td>DDPM 中训练稳定，目标简单</td><td>DDPM, Stable Diffusion 早期实现</td></tr>
                <tr><td>\(x_0\)</td><td>估计原始干净数据</td><td>直观，便于解释 denoiser</td><td>DDIM 推导、denoiser 表达</td></tr>
                <tr><td>score</td><td>\(\nabla_x\log p_t(x)\)</td><td>直接给出概率密度上升方向</td><td>Score-based SDE</td></tr>
                <tr><td>\(v\)</td><td>\(\alpha_t\epsilon-\sigma_t x_0\) 一类混合参数化</td><td>平衡高噪声和低噪声区域的数值稳定性</td><td>Imagen, Stable Diffusion v-parameterization</td></tr>
                <tr><td>\(u\)</td><td>Flow Matching 速度场</td><td>直接学习分布运输方向</td><td>Flow Matching, Rectified Flow</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="article-section">
          <h2>14. 初学者读论文时应该抓住什么？</h2>
          <p>读 DDPM 论文时，先抓住三件事：前向过程是固定高斯链；反向过程用神经网络参数化；训练目标从 VLB 简化成噪声 MSE。</p>
          <p>读 DDIM 论文时，关注它如何保持相同训练边缘分布，却改变采样路径，从而实现确定性和跳步采样。</p>
          <p>读 score-based SDE 论文时，关注 score 为什么能定义反向 SDE，以及 probability flow ODE 为什么能和 SDE 共享边缘分布。</p>
          <p>读 flow matching 论文时，关注 probability path、conditional vector field、marginal vector field 这三个概念，以及为什么回归速度场可以训练生成模型。</p>
          <p>如果读论文时迷路，可以反复回到这一句：生成模型是在学习一条从简单分布到复杂数据分布的运输路径。Diffusion、Score、ODE、Flow Matching 只是描述和学习这条路径的不同语言。</p>
        </section>

        <section class="article-section">
          <h2>15. 论文符号对照：看到不同记号不要慌</h2>
          <p>不同论文的符号经常不一样，但数学对象往往是同一个。读论文时先把符号翻译成本文的几个核心对象，会轻松很多。</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>你在论文里看到</th><th>常见含义</th><th>和本文的关系</th><th>读法</th></tr>
              </thead>
              <tbody>
                <tr><td>\(\sigma\)</td><td>噪声标准差</td><td>类似 \(\sqrt{1-\bar{\alpha}_t}\)</td><td>表示当前样本有多脏。</td></tr>
                <tr><td>\(\alpha(t),\sigma(t)\)</td><td>连续时间的信号系数和噪声系数</td><td>\(x_t=\alpha(t)x_0+\sigma(t)\epsilon\)</td><td>DDPM 闭式公式的连续版本。</td></tr>
                <tr><td>\(s_\theta(x,t)\)</td><td>score 网络</td><td>\(s_\theta\approx\nabla_x\log p_t(x)\)</td><td>告诉样本往哪里更像数据。</td></tr>
                <tr><td>\(D_\theta(x,\sigma)\)</td><td>denoiser</td><td>预测 \(x_0\) 的网络</td><td>EDM 等文章喜欢从去噪器角度写。</td></tr>
                <tr><td>\(\epsilon_\theta(x,t)\)</td><td>噪声预测网络</td><td>DDPM 的常用参数化</td><td>预测当初加进去的噪声。</td></tr>
                <tr><td>\(v_\theta(x,t)\)</td><td>速度或 velocity 参数化</td><td>可能是 diffusion 的 \(v\)-prediction，也可能是 flow 的速度场</td><td>要看上下文：是线性组合目标，还是 ODE 速度。</td></tr>
                <tr><td>probability flow ODE</td><td>和反向 SDE 共享边缘分布的确定性 ODE</td><td>由 score 构造速度</td><td>解释为什么不用随机噪声也能采样。</td></tr>
                <tr><td>conditional flow matching</td><td>用条件路径监督速度场</td><td>条件速度回归到边缘速度</td><td>类似“用条件信息训练，生成时只看当前点”。</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="article-section">
          <h2>16. 常见误解与纠正</h2>
          <h3>误解一：Diffusion 就是给图片加噪再去噪</h3>
          <p>这句话只说对了一半。加噪和去噪是训练和采样形式，但更本质的说法是：Diffusion 学习一族随时间变化的概率分布，以及从高噪声分布回到数据分布的方向。只把它理解成图像去噪，会很难理解 score、SDE、ODE 和 flow matching。</p>

          <h3>误解二：网络预测噪声，所以它只是学会了找噪声</h3>
          <p>预测噪声等价于预测当前 noisy sample 相对于干净数据的偏移方向，也等价于在已知尺度下预测条件 score。模型不是只在识别随机花纹，而是在所有噪声等级上学习“如何把样本拉回数据流形”。</p>

          <h3>误解三：DDPM 的反向过程就是把前向公式倒过来</h3>
          <p>前向公式可以从 \(x_0\) 得到 \(x_t\)，但反过来从 \(x_t\) 得到 \(x_0\) 是不适定问题。许多 \(x_0\) 都可能加噪到类似的 \(x_t\)。所以反向过程必须学习条件分布，而不是简单代数求逆。</p>

          <h3>误解四：SDE/ODE 很高级，可以先完全跳过</h3>
          <p>训练第一个 DDPM 时确实可以先不深究 SDE。但如果想读后续论文，比如 score-based generative modeling、EDM、flow matching、rectified flow、consistency model，连续时间语言几乎无法绕开。好消息是，它们的核心直觉仍然是本文这条路径：数据分布和噪声分布之间的运输。</p>

          <h3>误解五：Flow Matching 是 Diffusion 的完全替代品</h3>
          <p>更好的理解是：Flow Matching 和 Diffusion 都在描述从简单分布到数据分布的路径。Diffusion 常从 score 和反向随机过程出发；Flow Matching 常从速度场和连续性方程出发。它们有区别，也有可转换的部分。读新论文时，与其问“谁替代谁”，不如问“这篇论文选择了怎样的路径、怎样的参数化、怎样的训练监督”。</p>
        </section>

        <section class="article-section">
          <h2>17. 学完本文后，如何真正开始读论文？</h2>
          <p>推荐顺序不是按发表时间死排，而是按概念依赖来排。</p>
          <ol>
            <li><strong>DDPM。</strong>重点看前向闭式、后验高斯、VLB 分解、\(L_{\mathrm{simple}}\)。如果这些懂了，Diffusion 的离散基础就稳了。</li>
            <li><strong>DDIM。</strong>重点看为什么可以换采样路径，以及确定性采样如何从 \(\hat{x}_0\) 和 \(\epsilon_\theta\) 构造。</li>
            <li><strong>Score SDE。</strong>重点看 score、reverse-time SDE、probability flow ODE。不要被随机微分方程符号吓住，先抓住“score 修正反向漂移”。</li>
            <li><strong>EDM 或其他采样器文章。</strong>重点看噪声参数化、预条件、schedule 和数值求解器。它们经常是在改善“同一条路径怎么走得更稳更快”。</li>
            <li><strong>Flow Matching / Rectified Flow。</strong>重点看 probability path、velocity field、continuity equation，以及条件速度为什么能训练边缘速度。</li>
          </ol>
          <p>每读一篇论文，都可以用下面四个问题检查理解：</p>
          <ol>
            <li>这篇论文的数据端和噪声端分别是什么分布？</li>
            <li>它选择了什么路径连接这两个分布？</li>
            <li>神经网络输出的是什么对象：噪声、score、\(x_0\)、速度，还是别的参数化？</li>
            <li>训练损失为什么能监督这个对象？采样时又如何使用它？</li>
          </ol>
          <div class="insight-box">如果这四个问题都能回答，说明你已经不是在背公式，而是在理解生成模型的结构。</div>
        </section>

        <section class="article-section">
          <h2>18. 一个二维玩具例子：从数据到生成完整走一遍</h2>
          <p>为了确认我们不是只在符号层面理解 Diffusion，现在用一个二维玩具例子把整个流程走一遍。假设真实数据不是图片，而是二维平面上的点。真实分布由两个簇组成：左边一团点，右边一团点。你可以把它想成最简单的“数据流形”：真实样本大多落在两个区域，平面其他地方概率很低。</p>
          <p>训练集里每个样本是 \(x_0=(x^{(1)},x^{(2)})\)。我们并不知道真实密度 \(p_{\mathrm{data}}(x)\) 的解析式，只知道手上有一批样本。生成模型的目标是：以后不再依赖训练集，也能采样出新的点，而且这些点应该落在两个簇附近。</p>
          <p>第一步，定义前向加噪。对每个真实点 \(x_0\)，随机选一个时间 \(t\)，采样一份标准高斯噪声 \(\epsilon\)，然后合成：</p>
          <div class="equation">\[
            x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon.
          \]</div>
          <p>当 \(t\) 很小时，点仍然靠近原来的簇，只是有一点抖动；当 \(t\) 增大，两个簇开始变模糊；当 \(t\) 接近 \(T\)，所有点几乎都混成一个标准高斯云。这个过程不是模型学出来的，而是我们人为制造的训练题目。</p>
          <p>第二步，构造监督信号。注意我们在训练时知道 \(\epsilon\)，因为它就是刚刚采样出来的噪声。所以可以让网络看 \(x_t\) 和 \(t\)，输出 \(\epsilon_\theta(x_t,t)\)，并用真实 \(\epsilon\) 监督它。这个任务在二维例子里非常直观：如果一个点还在左边簇附近，网络要判断它偏离簇中心的噪声方向；如果一个点在两个簇之间，网络要根据统计规律判断它更可能来自哪个簇。</p>
          <p>第三步，理解网络学到的东西。低噪声时，两个簇还分得很清楚，模型学的是局部去噪：把点拉回附近簇。中等噪声时，两个簇开始重叠，模型学的是概率判断：同一个 \(x_t\) 可能来自左簇，也可能来自右簇，网络输出的是平均意义上的方向。高噪声时，几乎看不出原始簇结构，模型学的是从大范围噪声云中慢慢建立全局结构。</p>
          <p>第四步，生成。采样时我们不再有 \(x_0\)，只从 \(x_T\sim\mathcal{N}(0,I)\) 开始。网络根据当前 \(x_t\) 预测噪声，再用反向公式得到 \(x_{t-1}\)。一开始，点云只是无结构高斯；走几步后，点云开始往两个区域分裂；继续走，两个簇越来越清晰；最后得到的新点落在真实数据簇附近。</p>
          <p>这个二维例子可以解释为什么 Diffusion 适合高维图像。图像虽然复杂得多，但逻辑相同：真实图像分布是高维空间里有结构的区域；前向加噪逐渐抹掉结构；训练让模型在每个噪声等级上学习恢复结构的方向；采样从纯噪声开始，逐级把结构长出来。</p>
          <div class="insight-box">如果你能在脑子里想象二维点云从两个簇变成高斯云，再从高斯云分裂回两个簇，那么你已经抓住了 Diffusion 的主干。图像模型只是把这个二维直觉搬到极高维空间，并用 U-Net 或 Transformer 表达更复杂的方向场。</div>
        </section>

        <section class="article-section">
          <h2>19. 把 DDPM 论文的核心推导压成一条线</h2>
          <p>现在我们从论文视角重新串一次。这样做的目的不是重复，而是帮助你形成读论文时的“骨架”。以后看到新论文，你可以先判断它是在这条线的哪一步做改造。</p>
          <p><strong>第一步：目标是最大似然。</strong>我们希望模型给真实数据高概率，也就是最大化 \(\log p_\theta(x_0)\)。如果能直接计算这个量，就可以像普通概率模型一样训练。但 DDPM 的生成过程包含很多隐变量 \(x_1,\ldots,x_T\)，所以直接边缘化很难。</p>
          <div class="equation">\[
            p_\theta(x_0)=\int p_\theta(x_{0:T})dx_{1:T}.
          \]</div>
          <p><strong>第二步：引入一个已知的前向过程。</strong>我们定义 \(q(x_{1:T}|x_0)\)，它负责把数据加噪成高斯。这个过程不是要生成样本，而是为了构造一个可以训练的下界。它的好处是所有条件分布都是高斯，很多量可以解析计算。</p>
          <p><strong>第三步：用 Jensen 不等式得到 ELBO。</strong>把 \(q\) 乘进去再除出来，得到一个期望形式，然后把 \(\log\) 放进期望内部，就得到下界。这个步骤的意义是：我们把难算的 log marginal likelihood 变成了一个可以采样估计、可以分解的目标。</p>
          <p><strong>第四步：分解下界。</strong>下界分成三类项：最后的 \(x_T\) 是否像标准高斯，中间每一步反向分布是否接近真实后验，最后从 \(x_1\) 重建 \(x_0\) 是否合理。其中最重要的是中间 KL：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}
            \left(
            q(x_{t-1}|x_t,x_0)
            \,\|\,
            p_\theta(x_{t-1}|x_t)
            \right).
          \]</div>
          <p><strong>第五步：真实后验可以算。</strong>因为 \(q(x_t|x_{t-1})\) 和 \(q(x_{t-1}|x_0)\) 都是高斯，它们相乘仍然是高斯，所以 \(q(x_{t-1}|x_t,x_0)\) 可以精确写出均值和方差。这个地方是 DDPM 推导里最关键的代数部分。</p>
          <p><strong>第六步：模型后验也设成高斯。</strong>我们令：</p>
          <div class="equation">\[
            p_\theta(x_{t-1}|x_t)=
            \mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)).
          \]</div>
          <p>如果方差固定，训练主要就是让 \(\mu_\theta\) 接近真实后验均值 \(\tilde{\mu}_t\)。两个高斯的 KL 于是变成均值之间的平方误差。</p>
          <p><strong>第七步：用噪声预测参数化均值。</strong>真实后验均值依赖 \(x_0\)，但采样时我们没有 \(x_0\)。于是网络预测 \(\epsilon\)，再由 \(\epsilon_\theta\) 反推出 \(\hat{x}_0\)，再代入均值公式。这样模型只需要学一个统一任务：给定 \(x_t\) 和 \(t\)，预测当初加入的噪声。</p>
          <p><strong>第八步：得到简单 MSE。</strong>经过上述代换，复杂的 VLB 中间项变成带权的噪声 MSE。实践中进一步简化，去掉权重，得到 \(L_{\mathrm{simple}}\)。这就是为什么代码里看起来只有一行 MSE，但它背后其实连着最大似然、变分下界、后验高斯和均值参数化。</p>
          <p>所以，DDPM 不是“凭直觉训练一个去噪器”这么简单。它的理论闭环是完整的：概率模型定义了反向生成链；前向过程提供变分分布；ELBO 给出训练目标；高斯代数让目标可计算；噪声预测让训练稳定而简单。</p>
        </section>

        <section class="article-section">
          <h2>20. 图像模型里的网络结构：它不改变概率论，只负责表达函数</h2>
          <p>很多教程会很快跳到 U-Net、attention、time embedding、classifier-free guidance，这会让初学者误以为 Diffusion 的核心是某种神经网络结构。其实概率论部分和网络结构部分要分开看。</p>
          <p>从理论上说，网络只是在近似一个函数。这个函数可能是 \(\epsilon_\theta(x_t,t)\)，也可能是 \(s_\theta(x_t,t)\)、\(x_{\theta,0}(x_t,t)\) 或 \(v_\theta(x_t,t)\)。无论用 U-Net、DiT 还是其他架构，它们都在回答同一个问题：当前噪声样本 \(x_t\) 和噪声等级 \(t\) 给定后，应该输出什么方向信息。</p>
          <p>为什么图像 Diffusion 常用 U-Net？因为 U-Net 适合处理图像的多尺度结构。低层特征关心边缘和纹理，高层特征关心物体和语义；跳连可以把细节信息传回来。时间步 \(t\) 通常通过 time embedding 注入网络，让同一个网络知道当前面对的是低噪声、中噪声还是高噪声。</p>
          <p>attention 的作用是让远距离区域交换信息。生成图像时，一个物体的局部纹理要和整体布局协调，单纯卷积的局部感受野可能不够。加入 self-attention 或 cross-attention 后，模型可以更好地处理全局关系。文本到图像模型中的 cross-attention，则把文字条件注入图像生成过程。</p>
          <p>但是，网络结构再复杂，也没有改变本文推导出的核心关系。训练时仍然是采样 \(t\)、合成 \(x_t\)、预测目标、计算损失；采样时仍然是从噪声开始，反复调用网络得到方向，再更新样本。理解这一点很重要：架构是表达能力，概率路径是生成逻辑。</p>
          <div class="paper-note">读 Stable Diffusion 或 Latent Diffusion 时，也可以用同样框架理解。区别是扩散不直接发生在像素空间，而发生在 VAE 编码后的 latent space。这样计算更便宜，但前向加噪、噪声预测、反向采样这些理论骨架仍然存在。</div>
        </section>

        <section class="article-section">
          <h2>21. 自检：你是否真的能继续读论文了？</h2>
          <p>最后给自己做一次检查。不要只问“我是不是看完了”，而要问“我能不能不用看答案，把核心链条讲出来”。如果下面的问题能回答，说明基础理论已经基本打通。</p>
          <p>第一，你能否解释为什么 \(q(x_t|x_0)\) 可以一步采样？回答时应该说出独立高斯线性组合仍是高斯，信号系数累计成 \(\sqrt{\bar{\alpha}_t}\)，噪声方差累计成 \(1-\bar{\alpha}_t\)。如果只能背公式，还不算真正理解。</p>
          <p>第二，你能否解释为什么 \(q(x_{t-1}|x_t,x_0)\) 是高斯？回答时应该说出贝叶斯公式、Markov 性、两个高斯密度相乘、二次型配方、精度相加。这里是读 DDPM 论文最关键的门槛。</p>
          <p>第三，你能否解释为什么预测噪声可以训练生成模型？回答时不能只说“DDPM 就是这么做的”，而要说清楚：预测噪声可以反推出 \(\hat{x}_0\)，进而参数化反向均值；反向高斯 KL 在方差固定时变成均值 MSE；均值 MSE 又可以写成噪声 MSE。</p>
          <p>第四，你能否把 score 和噪声预测联系起来？应该能写出：</p>
          <div class="equation">\[
            \nabla_{x_t}\log q(x_t|x_0)
            =
            -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
          \]</div>
          <p>并且能解释这只是条件 score，真正生成需要边缘 score，而 denoising score matching 正是用条件监督学习边缘方向。</p>
          <p>第五，你能否解释 SDE、ODE、Flow Matching 的共同点？它们都在描述概率分布如何从噪声端移动到数据端。SDE 用随机过程和 score 修正反向漂移，probability flow ODE 用确定性速度保持同样边缘分布，Flow Matching 直接回归满足连续性方程的速度场。</p>
          <p>第六，你能否看到一篇新论文时，立刻定位它改变的是哪一层？有些论文改变噪声 schedule，有些改变网络输出参数化，有些改变采样器，有些把像素空间换成 latent space，有些把 score 语言换成 flow 语言。只要能把创新点放回“路径、目标、参数化、采样”这四个位置，就不会被新名词牵着走。</p>
          <p>第七，你能否把公式翻译成口头解释？比如 \(\bar{\alpha}_t\) 不是一个神秘符号，而是原始信号保留比例；\(\tilde{\mu}_t\) 不是凭空出现的均值，而是由 \(x_t\) 和 \(x_0\) 两个信息源按可靠性加权；score 不是抽象梯度，而是往高概率区域移动的方向。能做到这一点，说明公式已经变成理解，而不是记忆负担。</p>
          <p>第八，你能否解释为什么采样慢、为什么可以加速？慢是因为模型沿着很多离散噪声等级逐步移动，每一步只做局部修正；加速则来自更好的时间步选择、更稳定的 ODE/SDE 求解器、或者更直接的路径学习。这样看 DDIM、DPM-Solver、EDM、consistency model 时，就能明白它们大多是在改进“怎么走这条路”。</p>
          <p>如果这些问题还不能顺畅回答，就回到对应章节重新推一遍。Diffusion 理论最怕“看着都懂，合上全忘”。最有效的学习方式是自己在纸上从 \(x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon\) 开始，一路推到训练损失，再把它翻译成采样算法。真正掌握后，你应该能把每个公式都说成人话，也能把每句直觉重新写回公式。等你再读新论文时，就会知道作者究竟是在改路径、改目标、改采样器，还是改网络表达；这时阅读会从被动接受变成主动拆解，并形成自己的判断。</p>
        </section>

        <section class="article-section references">
          <h2>参考资料</h2>
          <ul>
            <li><a href="https://arxiv.org/abs/2006.11239" target="_blank" rel="noreferrer">Ho, Jain, Abbeel, Denoising Diffusion Probabilistic Models, 2020</a></li>
            <li><a href="https://arxiv.org/abs/2010.02502" target="_blank" rel="noreferrer">Song, Meng, Ermon, Denoising Diffusion Implicit Models, 2020</a></li>
            <li><a href="https://arxiv.org/abs/2011.13456" target="_blank" rel="noreferrer">Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, 2021</a></li>
            <li><a href="https://arxiv.org/abs/2102.09672" target="_blank" rel="noreferrer">Nichol, Dhariwal, Improved Denoising Diffusion Probabilistic Models, 2021</a></li>
            <li><a href="https://arxiv.org/abs/2206.00364" target="_blank" rel="noreferrer">Karras et al., Elucidating the Design Space of Diffusion-Based Generative Models, 2022</a></li>
            <li><a href="https://arxiv.org/abs/2207.12598" target="_blank" rel="noreferrer">Ho, Salimans, Classifier-Free Diffusion Guidance, 2022</a></li>
            <li><a href="https://arxiv.org/abs/2210.02747" target="_blank" rel="noreferrer">Lipman et al., Flow Matching for Generative Modeling, 2022</a></li>
            <li><a href="https://arxiv.org/abs/2112.10752" target="_blank" rel="noreferrer">Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, 2021</a></li>
            <li><a href="https://diffusionflow.github.io/" target="_blank" rel="noreferrer">Diffusion Meets Flow Matching</a></li>
            <li><a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/" target="_blank" rel="noreferrer">Lilian Weng, What are Diffusion Models?</a></li>
            <li><a href="https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html" target="_blank" rel="noreferrer">Cambridge MLG Blog, Flow Matching Guide and Code</a></li>
            <li><a href="https://arxiv.org/abs/2506.02070" target="_blank" rel="noreferrer">arXiv:2506.02070</a></li>
          </ul>
        </section>
      `,
    },
  ],
  tags: [
    "Diffusion Model",
    "Generative Model",
    "Machine Learning",
    "Probability",
    "DDPM",
    "Score Matching",
    "Flow Matching",
    "SDE",
    "ODE"
  ],
  categories: [{ name: "学习笔记", count: 1 }],
};
