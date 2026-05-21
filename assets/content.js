window.siteContent = {
  author: "MomenT",
  posts: [
    {
      slug: "diffusion-model-01-forward-process",
      title: "Diffusion Model 学习理解（一）：从生成建模到前向加噪",
      date: "2026-05-21",
      category: "学习笔记",
      summary:
        "从“为什么需要生成模型”开始，建立数据分布、噪声分布、逐步加噪、重参数化和闭式采样的完整直觉。重点推导 q(x_t | x_0) 为什么仍然是高斯分布。",
      tags: ["Diffusion Model", "Generative Model", "Machine Learning", "Probability"],
      cover: "cover-b",
      body: `
        <p class="lead">这一篇的目标不是马上训练模型，而是先回答一个更基础的问题：为什么 Diffusion Model 可以把“生成图片”这件事改写成“学会一步步去噪”？如果这个动机没有建立，后面的 DDPM、score matching、SDE、flow matching 都会变成一堆公式。</p>

        <section class="article-section">
          <h2>0. 先说清楚：生成模型到底在学什么？</h2>
          <p>我们手里有一批真实数据，比如图片、语音、文本 embedding 或某个二维玩具数据集。每一个样本可以抽象成一个向量 <span class="math">x ∈ R<sup>d</sup></span>。真实世界里存在一个未知的数据分布 <span class="math">p<sub>data</sub>(x)</span>，训练集里的样本可以看成从这个分布中抽出来的点。</p>
          <p>生成模型想做的事情，是构造一个模型分布 <span class="math">p<sub>θ</sub>(x)</span>，让它尽可能接近 <span class="math">p<sub>data</sub>(x)</span>。一旦接近，我们就可以从 <span class="math">p<sub>θ</sub></span> 中采样，得到新的样本。这里的“接近”可以有很多定义，比如最大似然、KL 散度、对抗损失、score matching 损失等。</p>
          <p>难点在于：真实数据分布很复杂。图像分布不是一个简单高斯，它可能集中在非常薄、非常弯曲的高维流形附近。直接写出 <span class="math">p<sub>θ</sub>(x)</span> 的概率密度很困难，直接从它采样也很困难。</p>
          <p>Diffusion 的核心想法很朴素：虽然真实数据很复杂，但标准高斯噪声非常简单。如果我们能找到一条路径，把真实数据分布慢慢变成高斯噪声；再训练一个模型学会沿着这条路径反向走回来，那么生成就变成了从噪声出发，一步步去噪。</p>
        </section>

        <section class="article-section">
          <h2>1. 前向过程：把数据慢慢加噪成标准高斯</h2>
          <p>DDPM 里的前向过程是一个人为设计的 Markov 链。给定真实样本 <span class="math">x<sub>0</sub></span>，我们定义一系列变量：</p>
          <div class="equation">x<sub>0</sub> → x<sub>1</sub> → x<sub>2</sub> → ... → x<sub>T</sub></div>
          <p>每一步只依赖上一步，也就是 Markov 性：</p>
          <div class="equation">q(x<sub>t</sub> | x<sub>t-1</sub>, x<sub>t-2</sub>, ..., x<sub>0</sub>) = q(x<sub>t</sub> | x<sub>t-1</sub>)</div>
          <p>这不是自然规律，而是我们主动选择的建模方式。为什么要这么选？因为如果每一步只依赖前一步，那么整个过程可以拆成很多个小的、可控的条件分布，数学和训练都会简单很多。</p>
          <p>DDPM 选择每一步加入一点高斯噪声：</p>
          <div class="equation">q(x<sub>t</sub> | x<sub>t-1</sub>) = N(x<sub>t</sub>; √α<sub>t</sub> x<sub>t-1</sub>, β<sub>t</sub>I)</div>
          <p>这里 <span class="math">β<sub>t</sub></span> 是第 <span class="math">t</span> 步加噪强度，通常是一个很小的正数；<span class="math">α<sub>t</sub> = 1 - β<sub>t</sub></span>。这个式子的采样写法是：</p>
          <div class="equation">x<sub>t</sub> = √α<sub>t</sub> x<sub>t-1</sub> + √β<sub>t</sub> ε<sub>t</sub>, &nbsp; ε<sub>t</sub> ~ N(0, I)</div>
          <p>为什么前面要乘 <span class="math">√α<sub>t</sub></span>，而不是直接写 <span class="math">x<sub>t</sub> = x<sub>t-1</sub> + noise</span>？因为我们希望方差保持可控。如果 <span class="math">x<sub>t-1</sub></span> 的每个维度大约方差为 1，那么 <span class="math">√α<sub>t</sub> x<sub>t-1</sub></span> 的方差约为 <span class="math">α<sub>t</sub></span>，噪声项 <span class="math">√β<sub>t</sub>ε</span> 的方差约为 <span class="math">β<sub>t</sub></span>。二者加起来大约是 <span class="math">α<sub>t</sub> + β<sub>t</sub> = 1</span>。这让数值尺度更稳定。</p>

          <figure class="article-figure chain-figure">
            <div class="chain-node clean">x<sub>0</sub><span>真实数据</span></div>
            <div class="chain-arrow">+</div>
            <div class="chain-node">x<sub>1</sub><span>少量噪声</span></div>
            <div class="chain-arrow">+</div>
            <div class="chain-node">x<sub>2</sub><span>更多噪声</span></div>
            <div class="chain-arrow">...</div>
            <div class="chain-node noise">x<sub>T</sub><span>近似高斯</span></div>
            <figcaption>前向过程是人为设计的加噪链。越往右，数据结构越弱，随机噪声越强。</figcaption>
          </figure>
        </section>

        <section class="article-section">
          <h2>2. 为什么可以一步从 x<sub>0</sub> 采样到 x<sub>t</sub>？</h2>
          <p>如果严格按定义采样，我们要从 <span class="math">x<sub>0</sub></span> 逐步算到 <span class="math">x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>t</sub></span>。但训练时我们经常随机抽一个时间步 <span class="math">t</span>，希望直接得到 <span class="math">x<sub>t</sub></span>。关键结论是：</p>
          <div class="equation">q(x<sub>t</sub> | x<sub>0</sub>) = N(x<sub>t</sub>; √ᾱ<sub>t</sub> x<sub>0</sub>, (1 - ᾱ<sub>t</sub>)I)</div>
          <p>其中 <span class="math">ᾱ<sub>t</sub> = ∏<sub>s=1</sub><sup>t</sup> α<sub>s</sub></span>。下面一步步推导，尽量不跳。</p>
          <h3>2.1 两步展开</h3>
          <p>先写第一步：</p>
          <div class="equation">x<sub>1</sub> = √α<sub>1</sub>x<sub>0</sub> + √β<sub>1</sub>ε<sub>1</sub></div>
          <p>第二步：</p>
          <div class="equation">x<sub>2</sub> = √α<sub>2</sub>x<sub>1</sub> + √β<sub>2</sub>ε<sub>2</sub></div>
          <p>把 <span class="math">x<sub>1</sub></span> 代进去：</p>
          <div class="equation">x<sub>2</sub> = √α<sub>2</sub>√α<sub>1</sub>x<sub>0</sub> + √α<sub>2</sub>√β<sub>1</sub>ε<sub>1</sub> + √β<sub>2</sub>ε<sub>2</sub></div>
          <p>前面的信号系数是 <span class="math">√(α<sub>1</sub>α<sub>2</sub>)</span>。噪声部分是两个独立高斯的线性组合。独立高斯线性组合仍然是高斯，方差是系数平方和：</p>
          <div class="equation">Var(noise) = α<sub>2</sub>β<sub>1</sub> + β<sub>2</sub></div>
          <p>因为 <span class="math">β<sub>1</sub> = 1 - α<sub>1</sub></span>，<span class="math">β<sub>2</sub> = 1 - α<sub>2</sub></span>，所以：</p>
          <div class="equation">α<sub>2</sub>(1 - α<sub>1</sub>) + (1 - α<sub>2</sub>) = 1 - α<sub>1</sub>α<sub>2</sub></div>
          <p>于是两步后：</p>
          <div class="equation">x<sub>2</sub> = √(α<sub>1</sub>α<sub>2</sub>)x<sub>0</sub> + √(1 - α<sub>1</sub>α<sub>2</sub>)ε, &nbsp; ε ~ N(0,I)</div>
          <p>这已经露出一般形式了。</p>
          <h3>2.2 一般 t 步</h3>
          <p>设 <span class="math">ᾱ<sub>t</sub> = α<sub>1</sub>α<sub>2</sub>...α<sub>t</sub></span>。如果第 <span class="math">t-1</span> 步已经满足：</p>
          <div class="equation">x<sub>t-1</sub> = √ᾱ<sub>t-1</sub>x<sub>0</sub> + √(1 - ᾱ<sub>t-1</sub>)ε</div>
          <p>代入第 <span class="math">t</span> 步：</p>
          <div class="equation">x<sub>t</sub> = √α<sub>t</sub>x<sub>t-1</sub> + √β<sub>t</sub>ε<sub>t</sub></div>
          <p>得到信号项：</p>
          <div class="equation">√α<sub>t</sub>√ᾱ<sub>t-1</sub>x<sub>0</sub> = √ᾱ<sub>t</sub>x<sub>0</sub></div>
          <p>噪声方差是：</p>
          <div class="equation">α<sub>t</sub>(1 - ᾱ<sub>t-1</sub>) + β<sub>t</sub> = 1 - α<sub>t</sub>ᾱ<sub>t-1</sub> = 1 - ᾱ<sub>t</sub></div>
          <p>所以结论成立：</p>
          <div class="equation">x<sub>t</sub> = √ᾱ<sub>t</sub>x<sub>0</sub> + √(1 - ᾱ<sub>t</sub>)ε, &nbsp; ε ~ N(0,I)</div>
          <p>这个式子非常重要。它告诉我们：任意时间步的 noisy sample 都可以由原图和一份标准高斯噪声直接合成，不需要真的跑完整条前向链。</p>
        </section>

        <section class="article-section">
          <h2>3. 可视化：t 越大，数据结构越弱</h2>
          <p>下面这个小可视化把二维数据点想象成“真实样本”。滑动时间步 <span class="math">t</span>，你会看到点云从有结构的两团数据逐渐变成接近圆形的高斯噪声。它对应的正是公式 <span class="math">x<sub>t</sub> = √ᾱ<sub>t</sub>x<sub>0</sub> + √(1-ᾱ<sub>t</sub>)ε</span>。</p>
          <div class="viz-card noise-viz" data-viz="noise">
            <canvas width="720" height="420" aria-label="前向加噪二维可视化"></canvas>
            <div class="viz-controls">
              <label>时间步 t <input type="range" min="0" max="100" value="0" /></label>
              <output>t = 0, 信号强</output>
            </div>
          </div>
          <p>从视觉上看，前向过程不是“把图片变差”这么简单，而是在构造一条从复杂数据分布到简单先验分布的桥。训练的任务，就是学习这座桥的反向方向。</p>
        </section>

        <section class="article-section">
          <h2>4. 这篇应该带走的直觉</h2>
          <ul>
            <li>生成模型的目标是从易采样的随机源出发，得到符合数据分布的新样本。</li>
            <li>Diffusion 先人为设计一个前向加噪过程，让复杂数据逐渐变成简单高斯。</li>
            <li>每一步加噪的形式选成高斯，是为了闭式计算、重参数化和训练稳定。</li>
            <li>核心公式 <span class="math">q(x<sub>t</sub>|x<sub>0</sub>)</span> 让我们可以随机抽时间步训练，而不用真的从 0 跑到 t。</li>
          </ul>
        </section>

        <section class="article-section references">
          <h2>参考资料</h2>
          <ul>
            <li><a href="https://arxiv.org/abs/2006.11239" target="_blank" rel="noreferrer">Ho, Jain, Abbeel, Denoising Diffusion Probabilistic Models, 2020</a></li>
            <li><a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/" target="_blank" rel="noreferrer">Lilian Weng, What are Diffusion Models?</a></li>
            <li><a href="https://diffusionflow.github.io/" target="_blank" rel="noreferrer">Diffusion Meets Flow Matching</a></li>
          </ul>
        </section>
      `,
    },
    {
      slug: "diffusion-model-02-ddpm-training",
      title: "Diffusion Model 学习理解（二）：反向去噪、DDPM 与训练目标",
      date: "2026-05-21",
      category: "学习笔记",
      summary:
        "从 q(x_{t-1}|x_t,x_0) 的后验推导开始，解释为什么模型要预测噪声 ε，为什么均方误差会变成 DDPM 的常用训练目标。",
      tags: ["Diffusion Model", "DDPM", "Variational Inference", "Probability"],
      cover: "cover-c",
      body: `
        <p class="lead">上一篇我们只做了前向加噪：把真实数据一步步变成噪声。这一篇进入 DDPM 的核心：如果前向过程是已知的，反向过程应该长什么样？模型到底学的是图片、噪声、score，还是某种条件均值？</p>

        <section class="article-section">
          <h2>1. 反向过程为什么困难？</h2>
          <p>前向过程 <span class="math">q(x<sub>t</sub>|x<sub>t-1</sub>)</span> 是我们设计的，所以完全知道。但生成时我们要从 <span class="math">x<sub>T</sub> ~ N(0,I)</span> 出发，逐步采样：</p>
          <div class="equation">p<sub>θ</sub>(x<sub>0:T</sub>) = p(x<sub>T</sub>) ∏<sub>t=1</sub><sup>T</sup> p<sub>θ</sub>(x<sub>t-1</sub>|x<sub>t</sub>)</div>
          <p>真正的反向条件分布 <span class="math">q(x<sub>t-1</sub>|x<sub>t</sub>)</span> 很难直接写，因为它要对所有可能的真实样本 <span class="math">x<sub>0</sub></span> 做积分。直观上，只看到一个带噪样本 <span class="math">x<sub>t</sub></span>，可能有很多干净样本都能加噪到它。</p>
          <p>DDPM 的技巧是：训练时我们知道 <span class="math">x<sub>0</sub></span>，所以先推导一个更容易的后验：</p>
          <div class="equation">q(x<sub>t-1</sub> | x<sub>t</sub>, x<sub>0</sub>)</div>
          <p>这个分布可以精确写成高斯。然后我们让神经网络学一个近似的反向高斯分布。</p>
        </section>

        <section class="article-section">
          <h2>2. 后验 q(x<sub>t-1</sub>|x<sub>t</sub>,x<sub>0</sub>) 的完整推导</h2>
          <p>我们要计算：</p>
          <div class="equation">q(x<sub>t-1</sub>|x<sub>t</sub>,x<sub>0</sub>) ∝ q(x<sub>t</sub>|x<sub>t-1</sub>) q(x<sub>t-1</sub>|x<sub>0</sub>)</div>
          <p>这个比例来自 Bayes 公式。分母 <span class="math">q(x<sub>t</sub>|x<sub>0</sub>)</span> 与 <span class="math">x<sub>t-1</sub></span> 无关，所以在求关于 <span class="math">x<sub>t-1</sub></span> 的分布形状时可以先忽略。</p>
          <p>两个因子分别是：</p>
          <div class="equation">q(x<sub>t</sub>|x<sub>t-1</sub>) = N(x<sub>t</sub>; √α<sub>t</sub>x<sub>t-1</sub>, β<sub>t</sub>I)</div>
          <div class="equation">q(x<sub>t-1</sub>|x<sub>0</sub>) = N(x<sub>t-1</sub>; √ᾱ<sub>t-1</sub>x<sub>0</sub>, (1-ᾱ<sub>t-1</sub>)I)</div>
          <p>为了看清楚均值，我们只看指数部分，并把 <span class="math">x = x<sub>t-1</sub></span> 当作未知量。第一个高斯的负二次项是：</p>
          <div class="equation">- 1/(2β<sub>t</sub>) ||x<sub>t</sub> - √α<sub>t</sub>x||<sup>2</sup></div>
          <p>展开里面的平方，只保留含 <span class="math">x</span> 的项：</p>
          <div class="equation">- 1/(2β<sub>t</sub>) (α<sub>t</sub>||x||<sup>2</sup> - 2√α<sub>t</sub>x<sub>t</sub><sup>T</sup>x)</div>
          <p>第二个高斯的负二次项是：</p>
          <div class="equation">- 1/[2(1-ᾱ<sub>t-1</sub>)] ||x - √ᾱ<sub>t-1</sub>x<sub>0</sub>||<sup>2</sup></div>
          <p>展开后含 <span class="math">x</span> 的项为：</p>
          <div class="equation">- 1/[2(1-ᾱ<sub>t-1</sub>)] (||x||<sup>2</sup> - 2√ᾱ<sub>t-1</sub>x<sub>0</sub><sup>T</sup>x)</div>
          <p>把两个负二次项合起来，可以写成：</p>
          <div class="equation">-1/2 [ A||x||<sup>2</sup> - 2b<sup>T</sup>x ] + const</div>
          <p>其中：</p>
          <div class="equation">A = α<sub>t</sub>/β<sub>t</sub> + 1/(1-ᾱ<sub>t-1</sub>)</div>
          <div class="equation">b = √α<sub>t</sub>x<sub>t</sub>/β<sub>t</sub> + √ᾱ<sub>t-1</sub>x<sub>0</sub>/(1-ᾱ<sub>t-1</sub>)</div>
          <p>高斯分布的指数标准形式是 <span class="math">-1/2 A ||x-μ||<sup>2</sup></span>。展开得到 <span class="math">-1/2[A||x||<sup>2</sup> - 2Aμ<sup>T</sup>x + ...]</span>。对比可知 <span class="math">Aμ = b</span>，所以 <span class="math">μ = b/A</span>。</p>
          <p>整理后可以得到 DDPM 常见写法：</p>
          <div class="equation">q(x<sub>t-1</sub>|x<sub>t</sub>,x<sub>0</sub>) = N(x<sub>t-1</sub>; μ̃<sub>t</sub>(x<sub>t</sub>,x<sub>0</sub>), β̃<sub>t</sub>I)</div>
          <div class="equation">β̃<sub>t</sub> = [(1-ᾱ<sub>t-1</sub>)/(1-ᾱ<sub>t</sub>)] β<sub>t</sub></div>
          <div class="equation">μ̃<sub>t</sub> = [√ᾱ<sub>t-1</sub>β<sub>t</sub>/(1-ᾱ<sub>t</sub>)]x<sub>0</sub> + [√α<sub>t</sub>(1-ᾱ<sub>t-1</sub>)/(1-ᾱ<sub>t</sub>)]x<sub>t</sub></div>
          <p>这个均值很有解释性：它是 <span class="math">x<sub>0</sub></span> 和 <span class="math">x<sub>t</sub></span> 的加权组合。也就是说，如果训练时知道原始干净图，那么最优的上一步估计可以精确算出来。</p>
        </section>

        <section class="article-section">
          <h2>3. 为什么模型通常预测噪声 ε？</h2>
          <p>训练时我们知道 <span class="math">x<sub>0</sub></span>，也知道随机采样的噪声 <span class="math">ε</span>，并通过：</p>
          <div class="equation">x<sub>t</sub> = √ᾱ<sub>t</sub>x<sub>0</sub> + √(1-ᾱ<sub>t</sub>)ε</div>
          <p>得到 <span class="math">x<sub>t</sub></span>。这条式子可以反过来解出 <span class="math">x<sub>0</sub></span>：</p>
          <div class="equation">x<sub>0</sub> = [x<sub>t</sub> - √(1-ᾱ<sub>t</sub>)ε] / √ᾱ<sub>t</sub></div>
          <p>如果神经网络能根据 <span class="math">x<sub>t</sub></span> 和 <span class="math">t</span> 预测噪声 <span class="math">ε</span>，我们就可以得到对 <span class="math">x<sub>0</sub></span> 的估计，再代回后验均值，完成一步反向采样。</p>
          <p>所以“预测噪声”不是玄学，而是一种参数化方式。预测 <span class="math">x<sub>0</sub></span>、预测 <span class="math">ε</span>、预测 score、预测 velocity，在数学上有很多互相转换关系。DDPM 论文发现预测噪声并使用简单 MSE 在实践中很好用。</p>
          <div class="equation">L<sub>simple</sub>(θ) = E<sub>x<sub>0</sub>,ε,t</sub> || ε - ε<sub>θ</sub>(x<sub>t</sub>,t) ||<sup>2</sup></div>
          <p>这个损失的训练过程可以拆成四步：</p>
          <ol>
            <li>从数据集中抽一张干净图 <span class="math">x<sub>0</sub></span>。</li>
            <li>随机抽一个时间步 <span class="math">t</span>。</li>
            <li>随机抽一份标准高斯噪声 <span class="math">ε</span>，合成 <span class="math">x<sub>t</sub></span>。</li>
            <li>让网络看 <span class="math">x<sub>t</sub></span> 和 <span class="math">t</span>，输出 <span class="math">ε<sub>θ</sub></span>，用 MSE 逼近真实噪声。</li>
          </ol>

          <figure class="article-figure training-figure">
            <div class="train-box">x<sub>0</sub><span>干净数据</span></div>
            <div class="train-plus">+</div>
            <div class="train-box">ε<span>真实噪声</span></div>
            <div class="train-arrow">→</div>
            <div class="train-box">x<sub>t</sub><span>带噪输入</span></div>
            <div class="train-arrow">→</div>
            <div class="train-box model-box">ε<sub>θ</sub><span>网络预测</span></div>
            <figcaption>训练时，目标不是直接“画图”，而是在每个噪声等级上学会识别加入了哪一份噪声。</figcaption>
          </figure>
        </section>

        <section class="article-section">
          <h2>4. 从 MSE 到反向采样</h2>
          <p>训练完成后，采样从 <span class="math">x<sub>T</sub> ~ N(0,I)</span> 开始。对每个时间步 <span class="math">t = T, T-1, ..., 1</span>：</p>
          <ol>
            <li>网络预测 <span class="math">ε<sub>θ</sub>(x<sub>t</sub>,t)</span>。</li>
            <li>用预测噪声估计干净样本 <span class="math">x̂<sub>0</sub></span>。</li>
            <li>把 <span class="math">x̂<sub>0</sub></span> 和当前 <span class="math">x<sub>t</sub></span> 代入后验均值公式。</li>
            <li>从对应高斯分布采样得到 <span class="math">x<sub>t-1</sub></span>。</li>
          </ol>
          <p>最朴素的 DDPM 采样会跑很多步，所以早期常见问题是慢。DDIM、DPM-Solver、概率流 ODE、flow matching 等路线，很多都可以理解成：在保持生成路径合理的前提下，尝试更少步、更稳定地从噪声走回数据。</p>
        </section>

        <section class="article-section">
          <h2>5. 这篇应该带走的直觉</h2>
          <ul>
            <li>真正困难的是反向条件分布 <span class="math">q(x<sub>t-1</sub>|x<sub>t</sub>)</span>，因为只看 noisy sample 不知道它来自哪个 clean sample。</li>
            <li>训练时知道 <span class="math">x<sub>0</sub></span>，所以可以推导精确后验 <span class="math">q(x<sub>t-1</sub>|x<sub>t</sub>,x<sub>0</sub>)</span>。</li>
            <li>预测噪声 ε 是一种方便稳定的参数化，它可以转换成对 <span class="math">x<sub>0</sub></span>、均值和 score 的估计。</li>
            <li>DDPM 的训练像是在所有噪声等级上做一个“噪声识别任务”。</li>
          </ul>
        </section>

        <section class="article-section references">
          <h2>参考资料</h2>
          <ul>
            <li><a href="https://arxiv.org/abs/2006.11239" target="_blank" rel="noreferrer">DDPM, Ho et al., 2020</a></li>
            <li><a href="https://arxiv.org/abs/2010.02502" target="_blank" rel="noreferrer">DDIM, Song, Meng, Ermon, 2020</a></li>
            <li><a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/" target="_blank" rel="noreferrer">Lilian Weng, What are Diffusion Models?</a></li>
          </ul>
        </section>
      `,
    },
    {
      slug: "diffusion-model-03-score-flow-matching",
      title: "Diffusion Model 学习理解（三）：Score、ODE/SDE 与 Flow Matching",
      date: "2026-05-21",
      category: "学习笔记",
      summary:
        "把 DDPM 的噪声预测和 score matching 联系起来，再解释 SDE、概率流 ODE、DDIM 与 Flow Matching 如何看成同一类生成路径的不同描述。",
      tags: ["Diffusion Model", "Score Matching", "Flow Matching", "SDE", "ODE"],
      cover: "cover-d",
      body: `
        <p class="lead">前两篇已经从离散 DDPM 的角度解释了加噪与去噪。这一篇把视角拉高：score 是什么？为什么 SDE 和 ODE 都能描述生成？Flow Matching 又为什么会和 Diffusion 联系起来？</p>

        <section class="article-section">
          <h2>1. Score 是概率密度的“上坡方向”</h2>
          <p>给定一个概率密度 <span class="math">p(x)</span>，它的 score 定义为：</p>
          <div class="equation">s(x) = ∇<sub>x</sub> log p(x)</div>
          <p>为什么不是直接用 <span class="math">∇p(x)</span>？因为 <span class="math">log p(x)</span> 的梯度有更好的尺度性质。直觉上，score 指向“概率密度增长最快”的方向。如果一个点落在数据分布边缘，score 会指向更像数据的区域。</p>
          <p>以一维标准高斯为例：</p>
          <div class="equation">p(x) = (1/√(2π)) exp(-x<sup>2</sup>/2)</div>
          <div class="equation">log p(x) = const - x<sup>2</sup>/2</div>
          <div class="equation">∇<sub>x</sub> log p(x) = -x</div>
          <p>这说明当 <span class="math">x</span> 在右边时，score 指向左；当 <span class="math">x</span> 在左边时，score 指向右；也就是把点拉回高斯中心。</p>

          <figure class="article-figure score-figure">
            <svg viewBox="0 0 720 260" role="img" aria-label="二维 score 场示意图">
              <defs>
                <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
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
            <figcaption>score 可以理解为把点推向高密度数据区域的方向。真实模型里这个方向由神经网络学习。</figcaption>
          </figure>
        </section>

        <section class="article-section">
          <h2>2. 噪声预测和 score 的关系</h2>
          <p>前向加噪的条件分布是：</p>
          <div class="equation">q(x<sub>t</sub>|x<sub>0</sub>) = N(x<sub>t</sub>; √ᾱ<sub>t</sub>x<sub>0</sub>, (1-ᾱ<sub>t</sub>)I)</div>
          <p>对 <span class="math">x<sub>t</sub></span> 求条件 log density 的梯度：</p>
          <div class="equation">∇<sub>x<sub>t</sub></sub> log q(x<sub>t</sub>|x<sub>0</sub>) = - [x<sub>t</sub> - √ᾱ<sub>t</sub>x<sub>0</sub>] / (1-ᾱ<sub>t</sub>)</div>
          <p>而根据重参数化：</p>
          <div class="equation">x<sub>t</sub> - √ᾱ<sub>t</sub>x<sub>0</sub> = √(1-ᾱ<sub>t</sub>) ε</div>
          <p>代入得到：</p>
          <div class="equation">∇<sub>x<sub>t</sub></sub> log q(x<sub>t</sub>|x<sub>0</sub>) = - ε / √(1-ᾱ<sub>t</sub>)</div>
          <p>这说明，如果网络能预测噪声 <span class="math">ε</span>，它也就在某种意义上学习了 noisy distribution 的 score。差别只是尺度因子。因此 DDPM 的噪声预测、score-based model 的 score 预测，在核心直觉上是紧密相关的。</p>
        </section>

        <section class="article-section">
          <h2>3. 从离散步到连续时间：SDE 视角</h2>
          <p>如果把时间步切得越来越细，离散加噪链可以变成连续时间随机微分方程。一般写成：</p>
          <div class="equation">dx = f(x,t)dt + g(t)dw</div>
          <p>其中 <span class="math">f(x,t)</span> 是漂移项，决定平均趋势；<span class="math">g(t)dw</span> 是随机噪声项，<span class="math">w</span> 是 Brownian motion。前向 SDE 把数据逐渐扩散成噪声。</p>
          <p>反向生成对应一个 reverse-time SDE。它的漂移项会用到 score：</p>
          <div class="equation">dx = [f(x,t) - g(t)<sup>2</sup>∇<sub>x</sub>log p<sub>t</sub>(x)]dt + g(t)d w_reverse</div>
          <p>这个式子的重点不是记符号，而是理解：反向过程需要知道当前 noisy distribution 的 score，因为 score 告诉我们应该往哪里走才能回到高密度数据区域。</p>
        </section>

        <section class="article-section">
          <h2>4. 概率流 ODE：没有随机噪声也能走回去</h2>
          <p>Score-based SDE 还有一个非常重要的对应物：probability flow ODE。它没有随机噪声项，但在每个时间点拥有和 SDE 相同的边缘分布 <span class="math">p<sub>t</sub>(x)</span>：</p>
          <div class="equation">dx = [f(x,t) - 1/2 g(t)<sup>2</sup>∇<sub>x</sub>log p<sub>t</sub>(x)]dt</div>
          <p>直觉上，SDE 是“带随机抖动的路径”，ODE 是“确定性的流”。二者都可以把噪声分布运输到数据分布，只是路径描述不同。DDIM 可以被理解为一种更确定性的采样路径，它允许用更少步生成样本。</p>
        </section>

        <section class="article-section">
          <h2>5. Flow Matching：直接学习速度场</h2>
          <p>Flow Matching 换了一个角度：与其学习 score 再构造反向 SDE/ODE，不如直接学习一个速度场 <span class="math">v<sub>θ</sub>(x,t)</span>，让样本点沿着 ODE 流动：</p>
          <div class="equation">dx/dt = v<sub>θ</sub>(x,t)</div>
          <p>如果我们设计一条从噪声样本 <span class="math">x<sub>0</sub></span> 到数据样本 <span class="math">x<sub>1</sub></span> 的条件路径，例如最简单的线性插值：</p>
          <div class="equation">x<sub>t</sub> = (1-t)x<sub>0</sub> + tx<sub>1</sub></div>
          <p>那么这条路径的真实速度就是：</p>
          <div class="equation">u<sub>t</sub> = d x<sub>t</sub>/dt = x<sub>1</sub> - x<sub>0</sub></div>
          <p>训练时让网络回归这个速度：</p>
          <div class="equation">L(θ) = E || v<sub>θ</sub>(x<sub>t</sub>,t) - u<sub>t</sub> ||<sup>2</sup></div>
          <p>这就是 Flow Matching 的基本味道。真实论文会使用更一般的 probability path 和条件速度场，但入门时先抓住这点：Diffusion 常常学习“往高密度走的方向”，Flow Matching 更直接学习“点应该以什么速度被运输”。</p>

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
          <h2>6. 一张表总结三种视角</h2>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>视角</th><th>模型学习什么</th><th>生成时怎么走</th><th>直觉</th></tr>
              </thead>
              <tbody>
                <tr><td>DDPM</td><td>噪声 ε 或均值</td><td>离散反向 Markov 链</td><td>一步步去噪</td></tr>
                <tr><td>Score SDE</td><td>score ∇log p<sub>t</sub>(x)</td><td>反向 SDE</td><td>沿高密度方向修正随机过程</td></tr>
                <tr><td>Probability Flow ODE</td><td>score 导出的 ODE 速度</td><td>确定性 ODE</td><td>不用随机项也保持边缘分布</td></tr>
                <tr><td>Flow Matching</td><td>速度场 v(x,t)</td><td>ODE 流</td><td>直接学习如何运输分布</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="article-section references">
          <h2>参考资料</h2>
          <ul>
            <li><a href="https://diffusionflow.github.io/" target="_blank" rel="noreferrer">Diffusion Meets Flow Matching</a></li>
            <li><a href="https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html" target="_blank" rel="noreferrer">Cambridge MLG Blog, Flow Matching Guide and Code</a></li>
            <li><a href="https://arxiv.org/abs/2210.02747" target="_blank" rel="noreferrer">Lipman et al., Flow Matching for Generative Modeling</a></li>
            <li><a href="https://arxiv.org/abs/2011.13456" target="_blank" rel="noreferrer">Song et al., Score-Based Generative Modeling through SDEs</a></li>
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
    "Variational Inference",
    "Score Matching",
    "Flow Matching",
    "SDE",
    "ODE",
  ],
  categories: [{ name: "学习笔记", count: 3 }],
};

