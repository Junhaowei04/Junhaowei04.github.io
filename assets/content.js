window.siteContent = {
  author: "MomenT",
  posts: [
    {
      slug: "vae-foundations",
      title: "VAE 基础理论：从潜变量模型到 ELBO、重参数化与生成",
      date: "2026-05-23",
      category: "学习笔记",
      summary:
        "一篇面向生成模型初学者的 VAE 长文。我们从普通 autoencoder 与潜变量生成模型讲起，完整推导 ELBO、变分后验、重参数化技巧、高斯 KL 闭式、训练目标、采样过程，并解释 posterior collapse、β-VAE 与 IWAE。",
      tags: [
        "VAE",
        "Generative Model",
        "Variational Inference",
        "ELBO",
        "Latent Variable Model",
        "KL Divergence",
        "Machine Learning"
      ],
      cover: "cover-e",
      body: String.raw`
        <p class="lead">VAE，也就是 Variational Autoencoder，是现代深度生成模型里非常重要的一条路线。它不像 GAN 那样直接训练一个生成器去骗判别器，也不像 Diffusion 那样设计一条逐步加噪再反向去噪的路径。VAE 的核心问题是：如果真实数据背后存在一个看不见的潜变量 \(z\)，我们能不能一边学会从 \(z\) 生成数据 \(x\)，一边从数据 \(x\) 推断它可能来自哪些 \(z\)？这篇文章会从零开始，把潜变量模型、变分推断、ELBO、重参数化技巧和 VAE 的训练目标完整连起来。</p>

        <section class="article-section">
          <h2>0. 先说清楚：VAE 想解决什么问题？</h2>
          <p>前两篇文章已经建立了一个基本观点：生成模型的目标是学习数据分布 \(p_{\mathrm{data}}(x)\)，并从学到的模型分布 \(p_\theta(x)\) 中采样。VAE 也不例外。它只是选择了一种很有结构的方式来表达 \(p_\theta(x)\)：引入一个低维潜变量 \(z\)。</p>
          <div class="insight-box">VAE 的基本想法是：先从一个简单先验 \(p(z)\) 中采样潜变量 \(z\)，再由神经网络参数化的解码器 \(p_\theta(x|z)\) 生成数据 \(x\)。</div>
          <p>这句话看起来简单，但里面藏着三个关键问题。</p>
          <ol>
            <li><strong>生成问题。</strong>给定 \(z\)，怎么生成 \(x\)？这由解码器 \(p_\theta(x|z)\) 完成。</li>
            <li><strong>推断问题。</strong>给定一个真实样本 \(x\)，它可能来自哪些 \(z\)？这涉及后验 \(p_\theta(z|x)\)。</li>
            <li><strong>训练问题。</strong>我们想最大化 \(\log p_\theta(x)\)，但 \(p_\theta(x)\) 要对所有可能 \(z\) 积分，通常难以直接计算。</li>
          </ol>
          <p>VAE 的名字里有 autoencoder，是因为它有 encoder 和 decoder；名字里有 variational，是因为它用变分推断近似难算的后验；它是 generative model，是因为训练完成后可以从先验 \(p(z)\) 采样并生成新数据。</p>
        </section>

        <section class="article-section">
          <h2>1. 普通 Autoencoder 为什么还不算完整生成模型？</h2>
          <p>普通 autoencoder 的结构非常直观。给一个输入 \(x\)，编码器把它压缩成隐表示 \(h\)，解码器再从 \(h\) 重建 \(x\)：</p>
          <div class="equation">\[
            h=f_\phi(x),\qquad \hat{x}=g_\theta(h).
          \]</div>
          <p>训练目标通常是让重建误差尽量小：</p>
          <div class="equation">\[
            \mathcal{L}_{\mathrm{AE}}
            =
            \|x-\hat{x}\|^2.
          \]</div>
          <p>它确实能学到压缩表示，但有一个生成模型视角下的大问题：训练后我们不知道该从哪里采样 \(h\)。编码器产生的 \(h\) 可能分布在潜空间中很不规则的区域。你随便从标准高斯里抽一个 \(h\)，解码器不一定知道怎么处理，因为训练时它可能从没见过这个区域。</p>
          <p>这就是普通 autoencoder 和生成模型之间的差距。生成模型不仅要会重建已有样本，还要规定一个可采样的潜空间分布，让我们能从这个分布出发生成新样本。VAE 的第一步，就是给潜变量 \(z\) 规定一个简单先验：</p>
          <div class="equation">\[
            p(z)=\mathcal{N}(0,I).
          \]</div>
          <p>这样生成过程就清楚了：先采样 \(z\sim\mathcal{N}(0,I)\)，再通过 decoder 得到 \(x\)。VAE 要做的，是让这个过程生成的数据分布接近真实数据分布。</p>
        </section>

        <section class="article-section">
          <h2>2. 潜变量生成模型：从 \(z\) 到 \(x\)</h2>
          <p>VAE 的生成模型可以写成两步：</p>
          <div class="equation">\[
            z\sim p(z),
            \qquad
            x\sim p_\theta(x|z).
          \]</div>
          <p>其中 \(p(z)\) 通常选标准高斯：</p>
          <div class="equation">\[
            p(z)=\mathcal{N}(z;0,I).
          \]</div>
          <p>\(p_\theta(x|z)\) 是解码器定义的条件分布。注意它不是简单的确定性函数 \(x=g_\theta(z)\)，而是概率分布。神经网络输出的是这个分布的参数。例如：</p>
          <ol>
            <li>如果 \(x\) 是二值图像，可以让 decoder 输出 Bernoulli 分布的概率。</li>
            <li>如果 \(x\) 是连续像素，可以让 decoder 输出高斯分布的均值，方差可以固定或一起学习。</li>
          </ol>
          <p>模型对一个样本 \(x\) 的边缘概率是把所有可能的 \(z\) 都考虑进去：</p>
          <div class="equation">\[
            p_\theta(x)
            =
            \int p_\theta(x|z)p(z)\,dz.
          \]</div>
          <p>这条公式是 VAE 的核心。它说：一个样本 \(x\) 的概率，不是由某一个潜变量决定的，而是所有潜变量路径共同贡献的总概率。每个 \(z\) 先按先验 \(p(z)\) 出现，再按 \(p_\theta(x|z)\) 生成 \(x\)。把这些可能性加起来，就是 \(p_\theta(x)\)。</p>

          <figure class="source-figure">
            <img src="https://lilianweng.github.io/posts/2018-08-12-vae/vae-graphical-model.png" alt="VAE 概率图模型" loading="lazy" />
            <figcaption>参考图：VAE 的概率图模型。实线部分表示生成模型 \(p_\theta(x,z)=p(z)p_\theta(x|z)\)，虚线部分表示用 encoder 近似后验 \(q_\phi(z|x)\)。图片来源：Lilian Weng, From Autoencoder to Beta-VAE。</figcaption>
          </figure>
        </section>

        <section class="article-section">
          <h2>2.5 一个具体例子：手写数字里的潜变量是什么？</h2>
          <p>如果只看公式，\(z\) 很容易显得抽象。我们用手写数字举例。假设 \(x\) 是一张手写数字图片。图片表面上是像素矩阵，但它背后可能由很多隐藏因素决定：数字类别、书写倾斜角度、笔画粗细、整体位置、个人书写风格、局部弯曲程度等。</p>
          <p>这些隐藏因素不一定真的一一对应到 \(z\) 的某一维，但 VAE 希望用一个连续潜变量 \(z\) 捕捉它们的主要变化。生成时，先采一个 \(z\)，这个 \(z\) 可以理解为“生成某个样本的隐含说明书”；decoder 再把这份说明书翻译成像素图片。</p>
          <p>如果 \(z\) 的某个方向学到了“笔画粗细”，那么沿这个方向移动，decoder 输出的数字可能会越来越粗；如果另一个方向学到了“倾斜角度”，沿那个方向移动，数字可能会逐渐旋转。真实模型中这种解释不一定完全干净，但这就是 latent representation 的直觉。</p>
          <p>为什么要让 \(z\) 连续？因为连续潜空间能表达平滑变化。两个相近的 \(z\) 解码后应该得到相似样本；两个样本的编码之间做插值，也应该得到自然过渡。普通 autoencoder 没有强约束时，潜空间可能到处是洞；VAE 用 KL 把每个样本的编码区域拉向同一个标准高斯先验，就是为了让潜空间更像一个可以连续行走的空间。</p>
          <div class="insight-box">VAE 中的 \(z\) 不是标签，也不是某个确定语义变量，而是模型为了解释数据变化而学习出的连续隐含原因。</div>
        </section>

        <section class="article-section">
          <h2>3. 为什么 \(\log p_\theta(x)\) 难算？</h2>
          <p>如果我们能直接计算 \(p_\theta(x)\)，训练就很简单：最大化训练集对数似然。</p>
          <div class="equation">\[
            \max_\theta
            \sum_{i=1}^{N}
            \log p_\theta(x^{(i)}).
          \]</div>
          <p>但 VAE 的 \(p_\theta(x)\) 是一个积分：</p>
          <div class="equation">\[
            p_\theta(x)=
            \int p_\theta(x|z)p(z)\,dz.
          \]</div>
          <p>如果 \(z\) 维度很低、decoder 很简单，这个积分也许还能算。但实际 VAE 里，\(z\) 往往是几十维、几百维，\(p_\theta(x|z)\) 又由神经网络定义。这个积分没有简单解析解，数值积分也会随维度爆炸。</p>
          <p>另一个难点是后验分布：</p>
          <div class="equation">\[
            p_\theta(z|x)
            =
            \frac{p_\theta(x|z)p(z)}{p_\theta(x)}.
          \]</div>
          <p>这个后验回答：“给定观察到的 \(x\)，哪些 \(z\) 更可能生成它？”它对理解数据非常重要，但分母 \(p_\theta(x)\) 正是难算的边缘似然。因此真实后验 \(p_\theta(z|x)\) 通常也难算。</p>
          <div class="insight-box">VAE 的关键动作：用一个可计算的近似后验 \(q_\phi(z|x)\) 替代难算的真实后验 \(p_\theta(z|x)\)，并由此推导出可优化的 ELBO。</div>
        </section>

        <section class="article-section">
          <h2>4. 变分后验 \(q_\phi(z|x)\)：encoder 的概率意义</h2>
          <p>VAE 的 encoder 不是普通 autoencoder 里的确定性编码器。它输出的是一个分布，通常写成：</p>
          <div class="equation">\[
            q_\phi(z|x).
          \]</div>
          <p>这个分布用来近似真实后验 \(p_\theta(z|x)\)。最常见的选择是对角高斯：</p>
          <div class="equation">\[
            q_\phi(z|x)
            =
            \mathcal{N}
            \left(
            z;
            \mu_\phi(x),
            \mathrm{diag}(\sigma_\phi^2(x))
            \right).
          \]</div>
          <p>也就是说，encoder 网络接收 \(x\)，输出两个向量：均值 \(\mu_\phi(x)\) 和标准差 \(\sigma_\phi(x)\)，它们共同定义一个潜变量分布。训练时，我们不是把 \(x\) 编码成一个固定点，而是编码成潜空间里的一个小区域。</p>
          <p>这样做有两个好处。</p>
          <ol>
            <li><strong>不确定性。</strong>同一个 \(x\) 可能对应多个合理潜变量，分布比单点更自然。</li>
            <li><strong>可生成性。</strong>通过 KL 正则，encoder 产生的潜变量分布会被拉向先验 \(p(z)=\mathcal{N}(0,I)\)，使潜空间更连续、更容易采样。</li>
          </ol>
          <p>所以 VAE 的 encoder 有一个很重要的身份：它是一个 inference network，也就是摊销变分推断里的近似后验网络。摊销的意思是，我们不为每个样本单独优化一个变分分布，而是训练一个共享网络 \(q_\phi(z|x)\)，让它看到任何 \(x\) 都能快速给出近似后验。</p>
        </section>

        <section class="article-section">
          <h2>4.5 真实后验、近似后验和两个误差来源</h2>
          <p>为了真正理解 VAE，必须分清三个分布。</p>
          <ol>
            <li><strong>先验 \(p(z)\)。</strong>生成前我们相信潜变量大致服从什么分布。最常见是 \(\mathcal{N}(0,I)\)。</li>
            <li><strong>真实后验 \(p_\theta(z|x)\)。</strong>在当前 decoder 参数 \(\theta\) 下，观察到 \(x\) 后，哪些 \(z\) 更可能生成它。</li>
            <li><strong>近似后验 \(q_\phi(z|x)\)。</strong>encoder 给出的可计算分布，用来近似真实后验。</li>
          </ol>
          <p>真实后验难算，是因为：</p>
          <div class="equation">\[
            p_\theta(z|x)
            =
            \frac{p_\theta(x|z)p(z)}
            {\int p_\theta(x|z)p(z)\,dz}.
          \]</div>
          <p>分母要对所有 \(z\) 积分。这个积分一难算，后验也跟着难算。变分推断的办法不是硬算它，而是在一个容易处理的分布族里找一个近似。比如我们限定 \(q_\phi(z|x)\) 只能是对角高斯，这样计算和采样都方便。</p>
          <p>但这样会带来两个误差来源。</p>
          <p><strong>第一，approximation gap。</strong>如果真实后验本身非常复杂，比如多峰、强相关、弯曲形状，而我们只允许 \(q_\phi(z|x)\) 是对角高斯，那么不管怎么调参数，它都可能无法完全贴近真实后验。这是分布族表达能力不足造成的。</p>
          <p><strong>第二，amortization gap。</strong>即使对每个样本都存在一个很好的高斯近似，我们也不是为每个样本单独优化一个最优高斯，而是训练一个共享 encoder 网络一次性输出所有样本的近似后验。共享网络带来效率，也可能带来额外误差。</p>
          <p>为什么还要这样做？因为它非常高效。传统变分推断可能需要对每个样本反复优化局部变分参数；VAE 训练好 encoder 后，给一个新样本 \(x\)，一次前向传播就能得到 \(q_\phi(z|x)\)。这就是摊销推断的价值。</p>
          <div class="insight-box">VAE 的 encoder 不是简单压缩器，而是在用一个神经网络快速近似每个样本的后验分布。</div>
        </section>

        <section class="article-section">
          <h2>5. ELBO 推导一：从 KL 分解看下界</h2>
          <p>现在推导 VAE 的训练目标。我们希望最大化 \(\log p_\theta(x)\)，但它难算。于是引入任意一个近似后验 \(q_\phi(z|x)\)。考虑它和真实后验之间的 KL 散度：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}
            \left(
            q_\phi(z|x)\,\|\,p_\theta(z|x)
            \right)
            =
            \mathbb{E}_{q_\phi(z|x)}
            \left[
            \log
            \frac{q_\phi(z|x)}{p_\theta(z|x)}
            \right].
          \]</div>
          <p>把贝叶斯公式代入：</p>
          <div class="equation">\[
            p_\theta(z|x)
            =
            \frac{p_\theta(x|z)p(z)}{p_\theta(x)}.
          \]</div>
          <p>于是：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}
            =
            \mathbb{E}_{q_\phi}
            \left[
            \log q_\phi(z|x)
            -
            \log p_\theta(x|z)
            -
            \log p(z)
            +
            \log p_\theta(x)
            \right].
          \]</div>
          <p>\(\log p_\theta(x)\) 与 \(z\) 无关，所以期望后仍然是它自己：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}
            =
            \log p_\theta(x)
            -
            \mathbb{E}_{q_\phi}
            [\log p_\theta(x|z)]
            +
            \mathbb{E}_{q_\phi}
            [\log q_\phi(z|x)-\log p(z)].
          \]</div>
          <p>最后一项就是 \(D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))\)。整理得到：</p>
          <div class="equation">\[
            \log p_\theta(x)
            =
            \underbrace{
            \mathbb{E}_{q_\phi(z|x)}
            [\log p_\theta(x|z)]
            -
            D_{\mathrm{KL}}
            (q_\phi(z|x)\|p(z))
            }_{\mathcal{L}(\theta,\phi;x)}
            +
            D_{\mathrm{KL}}
            (q_\phi(z|x)\|p_\theta(z|x)).
          \]</div>
          <p>由于 KL 散度总是非负，得到：</p>
          <div class="equation">\[
            \log p_\theta(x)
            \geq
            \mathcal{L}(\theta,\phi;x).
          \]</div>
          <p>这个 \(\mathcal{L}\) 就是 ELBO，Evidence Lower Bound，也就是边缘对数似然的下界。最大化 ELBO 有两层含义：一方面提高数据似然的下界，另一方面让近似后验 \(q_\phi(z|x)\) 接近真实后验 \(p_\theta(z|x)\)。</p>
        </section>

        <section class="article-section">
          <h2>6. ELBO 推导二：从 Jensen 不等式看下界</h2>
          <p>同一个 ELBO 还可以从 Jensen 不等式直接推出。先把 \(q_\phi(z|x)\) 乘进去再除出来：</p>
          <div class="equation">\[
            \log p_\theta(x)
            =
            \log
            \int
            q_\phi(z|x)
            \frac{p_\theta(x|z)p(z)}
            {q_\phi(z|x)}
            dz.
          \]</div>
          <p>把积分写成 \(q_\phi\) 下的期望：</p>
          <div class="equation">\[
            \log p_\theta(x)
            =
            \log
            \mathbb{E}_{q_\phi(z|x)}
            \left[
            \frac{p_\theta(x|z)p(z)}
            {q_\phi(z|x)}
            \right].
          \]</div>
          <p>因为 \(\log\) 是凹函数，Jensen 不等式给出：</p>
          <div class="equation">\[
            \log \mathbb{E}[Y]
            \geq
            \mathbb{E}[\log Y].
          \]</div>
          <p>所以：</p>
          <div class="equation">\[
            \log p_\theta(x)
            \geq
            \mathbb{E}_{q_\phi}
            \left[
            \log p_\theta(x|z)
            +
            \log p(z)
            -
            \log q_\phi(z|x)
            \right].
          \]</div>
          <p>拆开后就是：</p>
          <div class="equation">\[
            \mathcal{L}(\theta,\phi;x)
            =
            \mathbb{E}_{q_\phi(z|x)}
            [\log p_\theta(x|z)]
            -
            D_{\mathrm{KL}}
            (q_\phi(z|x)\|p(z)).
          \]</div>
          <p>这两个推导角度互补。KL 分解告诉我们 ELBO 和真实后验的差距是什么；Jensen 推导告诉我们为什么它一定是 \(\log p_\theta(x)\) 的下界。</p>
        </section>

        <section class="article-section">
          <h2>7. ELBO 的两个部分：重建项和正则项</h2>
          <p>ELBO 有两个部分：</p>
          <div class="equation">\[
            \mathcal{L}
            =
            \underbrace{
            \mathbb{E}_{q_\phi(z|x)}
            [\log p_\theta(x|z)]
            }_{\text{重建项}}
            -
            \underbrace{
            D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))
            }_{\text{正则项}}.
          \]</div>
          <p><strong>重建项</strong>要求从 \(x\) 推断出的 \(z\) 能够通过 decoder 重新解释 \(x\)。如果 \(p_\theta(x|z)\) 给真实 \(x\) 高概率，那么 \(\log p_\theta(x|z)\) 就大。</p>
          <p><strong>KL 正则项</strong>要求 encoder 给出的 \(q_\phi(z|x)\) 不要离先验 \(p(z)\) 太远。它防止每个样本把自己编码到潜空间里互不相干的孤岛上。只有潜空间整体接近标准高斯，我们才能在生成时从 \(p(z)\) 中随便采样并解码。</p>
          <p>训练时通常最小化负 ELBO：</p>
          <div class="equation">\[
            \mathcal{J}(\theta,\phi;x)
            =
            -
            \mathcal{L}
            =
            -
            \mathbb{E}_{q_\phi(z|x)}
            [\log p_\theta(x|z)]
            +
            D_{\mathrm{KL}}(q_\phi(z|x)\|p(z)).
          \]</div>
          <p>这就是代码里常见的形式：</p>
          <div class="equation">\[
            \mathrm{loss}
            =
            \mathrm{reconstruction\ loss}
            +
            \mathrm{KL\ loss}.
          \]</div>
          <p>但一定要记住，这不是两个随便拼起来的工程损失。它们是从最大化 \(\log p_\theta(x)\) 的下界严格推导出来的。</p>
        </section>

        <section class="article-section">
          <h2>7.5 从信息瓶颈理解 ELBO</h2>
          <p>ELBO 还有一个非常有用的直觉：它像是在做信息瓶颈。重建项希望 \(z\) 尽可能保留 \(x\) 的信息；KL 项希望 \(q_\phi(z|x)\) 不要离先验太远，也就是不要把每个样本编码得过于特殊。</p>
          <p>如果只有重建项，模型会倾向于把 \(x\) 的所有细节都塞进 \(z\)，像普通 autoencoder 一样追求完美重建。这样训练集重建可能很好，但潜空间可能变得支离破碎。生成时从 \(\mathcal{N}(0,I)\) 随机采样，可能落到 decoder 没见过的区域。</p>
          <p>如果 KL 项太强，\(q_\phi(z|x)\) 会被压得几乎等于 \(p(z)\)，也就是：</p>
          <div class="equation">\[
            q_\phi(z|x)\approx p(z).
          \]</div>
          <p>这时 \(z\) 几乎不再携带关于 \(x\) 的信息。decoder 接收到的潜变量和输入样本关系很弱，重建质量就会下降，甚至出现 posterior collapse。</p>
          <p>因此 VAE 的训练不是单纯追求重建，也不是单纯追求潜空间标准高斯，而是在两者之间找平衡：</p>
          <div class="equation">\[
            \text{好的 VAE}
            =
            \text{足够表达数据}
            +
            \text{潜空间足够规整}.
          \]</div>
          <p>这也解释了为什么 β-VAE 只改一个 \(\beta\) 就能显著改变模型行为。它本质上是在调信息瓶颈的强度：\(\beta\) 越大，瓶颈越窄；\(\beta\) 越小，瓶颈越宽。</p>
        </section>

        <section class="article-section">
          <h2>8. Decoder likelihood：为什么有时是 BCE，有时是 MSE？</h2>
          <p>重建项的具体形式取决于我们怎么定义 \(p_\theta(x|z)\)。</p>
          <p>如果 \(x\) 是二值图像，比如每个像素是 0 或 1，可以令：</p>
          <div class="equation">\[
            p_\theta(x|z)
            =
            \prod_{j}
            \mathrm{Bernoulli}
            (x_j;\pi_{\theta,j}(z)).
          \]</div>
          <p>于是负对数似然就是 binary cross entropy：</p>
          <div class="equation">\[
            -\log p_\theta(x|z)
            =
            -
            \sum_j
            \left[
            x_j\log \pi_{\theta,j}(z)
            +
            (1-x_j)
            \log(1-\pi_{\theta,j}(z))
            \right].
          \]</div>
          <p>如果 \(x\) 是连续变量，可以令：</p>
          <div class="equation">\[
            p_\theta(x|z)
            =
            \mathcal{N}
            (x;\mu_\theta(z),\sigma_x^2 I).
          \]</div>
          <p>如果 \(\sigma_x^2\) 固定，那么负对数似然和 MSE 只差常数与比例系数：</p>
          <div class="equation">\[
            -\log p_\theta(x|z)
            =
            \frac{1}{2\sigma_x^2}
            \|x-\mu_\theta(z)\|^2
            +
            \mathrm{const}.
          \]</div>
          <p>所以，BCE 和 MSE 不是凭感觉选的。它们对应不同的观测分布假设。读 VAE 代码时，看到 reconstruction loss，要先问：这里假设 \(p_\theta(x|z)\) 是什么分布？</p>
        </section>

        <section class="article-section">
          <h2>9. 重参数化技巧：随机采样如何反向传播？</h2>
          <p>VAE 训练时需要优化：</p>
          <div class="equation">\[
            \mathbb{E}_{q_\phi(z|x)}
            [\log p_\theta(x|z)].
          \]</div>
          <p>问题是 \(z\) 是从 \(q_\phi(z|x)\) 里采样出来的，而这个分布又依赖 \(\phi\)。如果直接写 \(z\sim q_\phi(z|x)\)，采样操作看起来像一个不可微的随机节点，梯度怎么传回 encoder？</p>
          <p>重参数化技巧的想法是：不要直接从依赖 \(\phi\) 的分布里采样，而是从一个不依赖 \(\phi\) 的标准噪声分布里采样，再用可微变换得到 \(z\)。对于对角高斯：</p>
          <div class="equation">\[
            z\sim
            \mathcal{N}
            (\mu_\phi(x),\mathrm{diag}(\sigma_\phi^2(x)))
          \]</div>
          <p>可以改写为：</p>
          <div class="equation">\[
            \epsilon\sim\mathcal{N}(0,I),
            \qquad
            z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
          \]</div>
          <p>这样随机性全部来自 \(\epsilon\)，而 \(\mu_\phi(x)\)、\(\sigma_\phi(x)\) 到 \(z\) 的路径是可微的。于是：</p>
          <div class="equation">\[
            \nabla_\phi
            \mathbb{E}_{q_\phi(z|x)}
            [f(z)]
            =
            \nabla_\phi
            \mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}
            [
            f(\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon)
            ].
          \]</div>
          <p>实际训练时用 Monte Carlo 近似这个期望，通常每个样本采一个 \(\epsilon\) 就够了：</p>
          <div class="equation">\[
            \mathbb{E}_{q_\phi(z|x)}
            [\log p_\theta(x|z)]
            \approx
            \log p_\theta
            (x|z),
            \qquad
            z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
          \]</div>

          <figure class="source-figure">
            <img src="https://lilianweng.github.io/posts/2018-08-12-vae/vae-gaussian.png" alt="高斯 VAE 架构与重参数化" loading="lazy" />
            <figcaption>参考图：高斯 VAE 中 encoder 输出 \(\mu\) 和 \(\sigma\)，通过重参数化采样 \(z\)，再由 decoder 重建 \(x\)。图片来源：Lilian Weng, From Autoencoder to Beta-VAE。</figcaption>
          </figure>
        </section>

        <section class="article-section">
          <h2>9.5 开山论文到底贡献了什么？SGVB 与 AEVB</h2>
          <p>Kingma 和 Welling 的 Auto-Encoding Variational Bayes 之所以是 VAE 开山之作，不只是因为它写出了 ELBO。ELBO 在变分推断里早就存在。真正关键的是：他们把神经网络生成模型、摊销推断、重参数化技巧和随机梯度优化组合成一个可训练框架。</p>
          <p>论文里常见几个术语，读的时候要对上。</p>
          <ol>
            <li><strong>recognition model。</strong>就是我们文章里说的 encoder 或 inference network，记作 \(q_\phi(z|x)\)。它负责从数据 \(x\) 近似推断潜变量 \(z\)。</li>
            <li><strong>generative model。</strong>就是 decoder 加先验，联合分布为 \(p_\theta(x,z)=p(z)p_\theta(x|z)\)。</li>
            <li><strong>SGVB estimator。</strong>Stochastic Gradient Variational Bayes estimator，指用重参数化和 Monte Carlo 样本估计 ELBO 梯度。</li>
            <li><strong>AEVB algorithm。</strong>Auto-Encoding Variational Bayes algorithm，指把 SGVB estimator 放进自动编码器式的 encoder-decoder 结构里端到端训练。</li>
          </ol>
          <p>对单个样本 \(x^{(i)}\)，ELBO 是：</p>
          <div class="equation">\[
            \mathcal{L}(\theta,\phi;x^{(i)})
            =
            \mathbb{E}_{q_\phi(z|x^{(i)})}
            [\log p_\theta(x^{(i)}|z)]
            -
            D_{\mathrm{KL}}
            (q_\phi(z|x^{(i)})\|p(z)).
          \]</div>
          <p>用 \(L\) 个 Monte Carlo 样本估计重建期望，得到：</p>
          <div class="equation">\[
            \widetilde{\mathcal{L}}(\theta,\phi;x^{(i)})
            =
            \frac{1}{L}
            \sum_{\ell=1}^{L}
            \log p_\theta(x^{(i)}|z^{(i,\ell)})
            -
            D_{\mathrm{KL}}
            (q_\phi(z|x^{(i)})\|p(z)),
          \]</div>
          <div class="equation">\[
            z^{(i,\ell)}
            =
            \mu_\phi(x^{(i)})
            +
            \sigma_\phi(x^{(i)})\odot
            \epsilon^{(\ell)},
            \qquad
            \epsilon^{(\ell)}\sim\mathcal{N}(0,I).
          \]</div>
          <p>在整个数据集上，目标是所有样本 ELBO 的和：</p>
          <div class="equation">\[
            \mathcal{L}(\theta,\phi;X)
            =
            \sum_{i=1}^{N}
            \mathcal{L}(\theta,\phi;x^{(i)}).
          \]</div>
          <p>实际训练用 minibatch。若 minibatch 大小为 \(M\)，可以用：</p>
          <div class="equation">\[
            \mathcal{L}(\theta,\phi;X)
            \approx
            \frac{N}{M}
            \sum_{i=1}^{M}
            \widetilde{\mathcal{L}}(\theta,\phi;x^{(i)}).
          \]</div>
          <p>这就是 VAE 能在大数据和深度网络上训练的关键：不需要对每个数据点做昂贵的局部变分优化，也不需要精确计算边缘似然。每次只取一个 minibatch，用 encoder 给出 \(q_\phi(z|x)\)，重参数化采样 \(z\)，decoder 计算重建概率，再加上 KL 闭式项，就能用反向传播更新参数。</p>
          <div class="insight-box">读开山论文时，可以把 AEVB 理解成一句话：用一个 recognition network 产生近似后验，用重参数化得到低方差梯度，用随机梯度最大化 ELBO。</div>
        </section>

        <section class="article-section">
          <h2>9.6 为什么重参数化比普通梯度估计更关键？</h2>
          <p>如果没有重参数化技巧，我们仍然可以用一种通用公式对期望求梯度。设：</p>
          <div class="equation">\[
            \mathcal{F}(\phi)
            =
            \mathbb{E}_{q_\phi(z|x)}[f(z)].
          \]</div>
          <p>score-function estimator 给出：</p>
          <div class="equation">\[
            \nabla_\phi \mathcal{F}(\phi)
            =
            \mathbb{E}_{q_\phi(z|x)}
            \left[
            f(z)\nabla_\phi\log q_\phi(z|x)
            \right].
          \]</div>
          <p>这个公式很通用，离散变量也能用，但它通常方差很大。直觉上，它只根据“这次采到的 \(z\) 让 \(f(z)\) 大不大”来调整分布参数，随机性很强，需要很多样本才能稳定。</p>
          <p>重参数化技巧走的是另一条路。把随机变量写成：</p>
          <div class="equation">\[
            z=g_\phi(\epsilon,x),
            \qquad
            \epsilon\sim p(\epsilon),
          \]</div>
          <p>其中 \(p(\epsilon)\) 不依赖 \(\phi\)。于是：</p>
          <div class="equation">\[
            \nabla_\phi
            \mathbb{E}_{q_\phi(z|x)}[f(z)]
            =
            \nabla_\phi
            \mathbb{E}_{p(\epsilon)}
            [f(g_\phi(\epsilon,x))].
          \]</div>
          <p>因为 \(g_\phi\) 是可微函数，梯度可以沿着 \(f\) 对 \(z\) 的变化、再沿着 \(z\) 对 \(\phi\) 的变化传回来。这常被称为 pathwise gradient estimator。它利用了函数的局部导数信息，通常比 score-function estimator 方差更低。</p>
          <p>这就是为什么 VAE 开山论文如此重要：它不是只说“我们有个下界”，而是给出了一个在深度网络里稳定优化这个下界的方法。没有这个低方差梯度估计，ELBO 可能只是漂亮公式；有了重参数化，它才变成可训练模型。</p>
        </section>

        <section class="article-section">
          <h2>10. 高斯 KL 的闭式公式</h2>
          <p>VAE 最常见设定是：</p>
          <div class="equation">\[
            q_\phi(z|x)
            =
            \mathcal{N}
            (z;\mu,\mathrm{diag}(\sigma^2)),
            \qquad
            p(z)=\mathcal{N}(0,I).
          \]</div>
          <p>此时 KL 有闭式解：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}
            (q_\phi(z|x)\|p(z))
            =
            \frac{1}{2}
            \sum_{j=1}^{d}
            \left(
            \mu_j^2+\sigma_j^2-\log\sigma_j^2-1
            \right).
          \]</div>
          <p>我们推一维情形。设 \(q(z)=\mathcal{N}(\mu,\sigma^2)\)，\(p(z)=\mathcal{N}(0,1)\)。</p>
          <div class="equation">\[
            D_{\mathrm{KL}}(q\|p)
            =
            \mathbb{E}_q[\log q(z)-\log p(z)].
          \]</div>
          <p>两个 log density 分别是：</p>
          <div class="equation">\[
            \log q(z)
            =
            -\frac{1}{2}\log(2\pi\sigma^2)
            -
            \frac{(z-\mu)^2}{2\sigma^2},
          \]</div>
          <div class="equation">\[
            \log p(z)
            =
            -\frac{1}{2}\log(2\pi)
            -
            \frac{z^2}{2}.
          \]</div>
          <p>相减再对 \(q\) 取期望：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}(q\|p)
            =
            -\frac{1}{2}\log\sigma^2
            -
            \frac{1}{2}
            \mathbb{E}_q
            \left[
            \frac{(z-\mu)^2}{\sigma^2}
            \right]
            +
            \frac{1}{2}
            \mathbb{E}_q[z^2].
          \]</div>
          <p>因为 \(z\sim\mathcal{N}(\mu,\sigma^2)\)，所以：</p>
          <div class="equation">\[
            \mathbb{E}_q[(z-\mu)^2]=\sigma^2,
            \qquad
            \mathbb{E}_q[z^2]=\mu^2+\sigma^2.
          \]</div>
          <p>代入得到：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}(q\|p)
            =
            \frac{1}{2}
            \left(
            \mu^2+\sigma^2-\log\sigma^2-1
            \right).
          \]</div>
          <p>多维对角高斯就是把每一维相加。这个 KL 项会推动 \(\mu_j\) 接近 0，推动 \(\sigma_j^2\) 接近 1。重建项则希望 \(z\) 携带足够信息来还原 \(x\)。VAE 的潜空间质量，正是在这两股力量之间形成的。</p>
        </section>

        <section class="article-section">
          <h2>11. VAE 训练算法：从公式到代码</h2>
          <p>对一个 minibatch，VAE 的训练流程可以写成：</p>
          <ol>
            <li>输入样本 \(x\)。</li>
            <li>encoder 输出 \(\mu_\phi(x)\) 和 \(\log\sigma_\phi^2(x)\)。</li>
            <li>采样 \(\epsilon\sim\mathcal{N}(0,I)\)。</li>
            <li>用重参数化得到 \(z=\mu+\sigma\odot\epsilon\)。</li>
            <li>decoder 根据 \(z\) 输出 \(p_\theta(x|z)\) 的参数。</li>
            <li>计算重建损失和 KL 损失。</li>
            <li>最小化负 ELBO，更新 \(\theta,\phi\)。</li>
          </ol>
          <p>常见实现会让网络输出 log variance，也就是 \(\log\sigma^2\)，记作 \(\mathrm{logvar}\)。这样可以避免直接预测方差时出现负数。标准差由：</p>
          <div class="equation">\[
            \sigma=\exp\left(\frac{1}{2}\mathrm{logvar}\right)
          \]</div>
          <p>得到。KL 项可以写成代码里非常常见的形式：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}
            =
            -\frac{1}{2}
            \sum_j
            \left(
            1+\mathrm{logvar}_j-\mu_j^2-\exp(\mathrm{logvar}_j)
            \right).
          \]</div>
          <p>这个式子和上一节的：</p>
          <div class="equation">\[
            \frac{1}{2}
            \sum_j
            \left(
            \mu_j^2+\sigma_j^2-\log\sigma_j^2-1
            \right)
          \]</div>
          <p>完全一样，只是把 \(\sigma_j^2=\exp(\mathrm{logvar}_j)\) 代进去并整理了符号。</p>
        </section>

        <section class="article-section">
          <h2>11.5 代码变量和数学符号怎么对应？</h2>
          <p>初学者写 VAE 代码时经常卡在变量名上。下面把数学对象翻译成常见代码对象。</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>数学符号</th><th>代码里常见名字</th><th>含义</th></tr>
              </thead>
              <tbody>
                <tr><td>\(x\)</td><td>input, x</td><td>输入样本，例如一张图片。</td></tr>
                <tr><td>\(\mu_\phi(x)\)</td><td>mu</td><td>encoder 输出的潜变量均值。</td></tr>
                <tr><td>\(\log\sigma_\phi^2(x)\)</td><td>logvar</td><td>encoder 输出的潜变量 log variance。</td></tr>
                <tr><td>\(\epsilon\)</td><td>eps</td><td>标准高斯噪声，和网络参数无关。</td></tr>
                <tr><td>\(z\)</td><td>z</td><td>通过重参数化得到的潜变量样本。</td></tr>
                <tr><td>\(p_\theta(x|z)\)</td><td>decoder output</td><td>decoder 定义的观测分布参数。</td></tr>
                <tr><td>\(-\mathbb{E}\log p_\theta(x|z)\)</td><td>recon_loss</td><td>重建负对数似然，可能是 BCE 或 MSE。</td></tr>
                <tr><td>\(D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))\)</td><td>kl_loss</td><td>把近似后验拉向先验的正则项。</td></tr>
              </tbody>
            </table>
          </div>
          <p>一次训练前向传播可以写成概念伪代码：</p>
          <div class="equation">\[
            \mu,\mathrm{logvar}
            =
            \mathrm{Encoder}_\phi(x),
          \]</div>
          <div class="equation">\[
            \epsilon\sim\mathcal{N}(0,I),
            \qquad
            z=\mu+\exp(0.5\mathrm{logvar})\odot\epsilon,
          \]</div>
          <div class="equation">\[
            \hat{x}
            =
            \mathrm{Decoder}_\theta(z),
          \]</div>
          <div class="equation">\[
            \mathrm{loss}
            =
            \mathrm{recon\_loss}(x,\hat{x})
            +
            \mathrm{kl\_loss}(\mu,\mathrm{logvar}).
          \]</div>
          <p>这四行看起来像普通神经网络训练，但每一行都对应前面的概率推导。encoder 不是随便输出两个向量，而是在参数化 \(q_\phi(z|x)\)；采样不是普通噪声增强，而是在估计 ELBO 的期望；decoder 输出不是简单图片，而是在参数化 \(p_\theta(x|z)\)；loss 也不是经验拼接，而是负 ELBO。</p>
        </section>

        <section class="article-section">
          <h2>12. 训练完成后如何生成？</h2>
          <p>VAE 生成新样本时不需要 encoder。流程非常简单：</p>
          <ol>
            <li>从先验采样 \(z\sim\mathcal{N}(0,I)\)。</li>
            <li>把 \(z\) 输入 decoder。</li>
            <li>从 \(p_\theta(x|z)\) 中采样，或者直接取 decoder 输出的均值作为生成结果。</li>
          </ol>
          <p>这点和训练过程不同。训练时 encoder 用来构造近似后验 \(q_\phi(z|x)\)，帮助优化 ELBO；生成时我们没有输入样本 \(x\)，所以直接从先验开始。</p>
          <p>VAE 还可以做 latent interpolation。给两张图 \(x_a,x_b\)，用 encoder 得到两个潜空间均值 \(\mu_a,\mu_b\)，然后在二者之间插值：</p>
          <div class="equation">\[
            z_\lambda=(1-\lambda)\mu_a+\lambda\mu_b,
            \qquad
            0\leq \lambda\leq 1.
          \]</div>
          <p>再把 \(z_\lambda\) 输入 decoder，就能观察生成样本如何连续变化。如果潜空间被 KL 正则得比较平滑，插值通常也会比较自然。这正是 VAE 比普通 autoencoder 更适合作为生成模型的原因之一。</p>
        </section>

        <section class="article-section">
          <h2>12.5 聚合后验：为什么只约束每个 \(q_\phi(z|x)\) 还不够？</h2>
          <p>前面一直在说让 \(q_\phi(z|x)\) 接近先验 \(p(z)\)。但从整体生成角度，还可以看另一个对象：聚合后验，也叫 aggregated posterior。</p>
          <div class="equation">\[
            q_\phi(z)
            =
            \int q_\phi(z|x)p_{\mathrm{data}}(x)\,dx.
          \]</div>
          <p>它表示：如果先从真实数据分布里抽一个 \(x\)，再用 encoder 抽一个 \(z\)，最终得到的 \(z\) 在潜空间里整体服从什么分布。训练时 decoder 看到的 \(z\)，大体来自这个聚合后验；生成时 decoder 看到的 \(z\)，来自先验 \(p(z)\)。</p>
          <p>如果 \(q_\phi(z)\) 和 \(p(z)\) 差得很远，就会出现 train-test mismatch。训练时 decoder 习惯了某些潜空间区域，生成时却从标准高斯的其他区域采样，结果可能变差。</p>
          <p>VAE 的 KL 项是逐样本的：</p>
          <div class="equation">\[
            \mathbb{E}_{p_{\mathrm{data}}(x)}
            D_{\mathrm{KL}}
            (q_\phi(z|x)\|p(z)).
          \]</div>
          <p>它会间接推动聚合后验靠近先验，但这个约束有时太强，有时又不够精细。太强时，每个样本的后验都被压向同一个标准高斯，潜变量信息不足；不够精细时，聚合后验可能有复杂结构，而简单标准高斯先验无法匹配。</p>
          <p>这也是很多 VAE 后续工作会改先验的原因。例如使用 mixture prior、VampPrior、normalizing flow prior 等，让 \(p(z)\) 更接近模型实际学到的潜空间结构。读这些论文时，核心问题就是：标准高斯先验是不是太简单？如果太简单，我们该怎么让生成时采样的 \(z\) 更接近训练时 decoder 熟悉的 \(z\)？</p>
        </section>

        <section class="article-section">
          <h2>13. 为什么 VAE 有时会生成模糊图像？</h2>
          <p>很多人第一次训练 VAE，会发现生成图像偏模糊。这不是偶然现象，和目标函数有关。</p>
          <p>如果 decoder likelihood 设成固定方差高斯，那么最大化 \(\log p_\theta(x|z)\) 等价于最小化 MSE。MSE 在面对多种可能输出时，容易倾向于平均解。比如同一个潜变量附近可能对应多种合理边缘或纹理，取平均就会变模糊。</p>
          <p>另一个原因是 KL 正则会限制潜变量携带的信息。KL 太强时，\(q_\phi(z|x)\) 被迫非常接近 \(p(z)\)，不同样本的编码差别变小，decoder 得到的信息不足，也可能生成较模糊的结果。</p>
          <p>更强的 decoder 也可能导致 posterior collapse。此时 decoder 自己就能很好地建模数据，不太需要看 \(z\)，于是 \(q_\phi(z|x)\) 退化到接近 \(p(z)\)，潜变量被忽略。这个问题在文本 VAE 或很强自回归 decoder 中尤其常见。</p>
          <div class="insight-box">VAE 的模糊不是“模型坏了”这么简单，而是 likelihood 选择、KL 强度、decoder 表达能力和潜变量使用方式共同作用的结果。</div>
        </section>

        <section class="article-section">
          <h2>13.5 Posterior Collapse 更细地看</h2>
          <p>posterior collapse 指的是近似后验退化到先验：</p>
          <div class="equation">\[
            q_\phi(z|x)\approx p(z).
          \]</div>
          <p>此时 KL 项接近 0，看起来 loss 里 KL 很漂亮，但这反而可能说明模型没有使用潜变量。因为如果 \(q_\phi(z|x)\) 和 \(p(z)\) 几乎一样，那么 \(z\) 中几乎没有关于 \(x\) 的信息。decoder 生成时主要靠自己的条件建模能力，而不是靠潜变量传来的信息。</p>
          <p>为什么会发生？一个直觉解释是：ELBO 同时奖励重建、惩罚偏离先验。如果 decoder 很强，它即使不看 \(z\) 也能把训练数据建模得不错；那 encoder 继续把信息塞进 \(z\) 只会增加 KL 惩罚。优化器可能找到一个“省事”的方案：让 \(q_\phi(z|x)\) 贴近 \(p(z)\)，把潜变量关掉。</p>
          <p>常见缓解方法包括：</p>
          <ol>
            <li><strong>KL annealing。</strong>训练早期让 KL 权重较小，让模型先学会使用 \(z\)，之后再逐渐增大 KL 权重。</li>
            <li><strong>free bits。</strong>允许每个潜变量维度保留一定信息量，不让 KL 太快压到 0。</li>
            <li><strong>限制 decoder 过强。</strong>如果 decoder 可以完全绕过 \(z\)，潜变量就容易失去作用。</li>
            <li><strong>更灵活的后验或先验。</strong>例如 normalizing flow posterior、VampPrior 等方法，让近似分布族更匹配真实后验。</li>
          </ol>
          <p>读实验曲线时，如果看到 KL loss 快速降到接近 0，同时重建还不错，不要立刻高兴。你需要检查 latent 是否真的携带信息，比如做 latent traversal、互信息估计，或者观察随机采样和插值效果。</p>
        </section>

        <section class="article-section">
          <h2>14. β-VAE、IWAE 和常见扩展</h2>
          <p><strong>β-VAE</strong>把 ELBO 中的 KL 项乘上一个系数：</p>
          <div class="equation">\[
            \mathcal{L}_{\beta}
            =
            \mathbb{E}_{q_\phi(z|x)}
            [\log p_\theta(x|z)]
            -
            \beta
            D_{\mathrm{KL}}(q_\phi(z|x)\|p(z)).
          \]</div>
          <p>当 \(\beta>1\) 时，模型更强调潜空间接近先验，可能得到更 disentangled 的表示，但重建质量可能下降。当 \(\beta<1\) 时，潜变量可以携带更多信息，重建可能更好，但潜空间可能不够规整。</p>
          <p>这里的 disentanglement 指的是：潜变量的不同维度尽量对应数据变化中的不同因素。例如在理想情况下，一维控制旋转，一维控制粗细，一维控制位置。现实中这种对应不一定完美，也不一定自动出现，但 β-VAE 的动机就是通过更强的信息瓶颈，让模型倾向于学习更独立、更可解释的因素。</p>
          <p>为什么更大的 \(\beta\) 可能帮助 disentanglement？一个直觉是：当瓶颈更窄时，模型不能随便把所有细节都塞进 \(z\)，只能保留对重建最有用、最稳定的变化因素。这样潜变量可能更有结构。但代价也明显：如果瓶颈太窄，细节丢失，重建和生成质量都会下降。</p>
          <p>所以 β-VAE 不是“β 越大越好”。它是在三个目标之间权衡：重建质量、生成质量、表示可解释性。读相关论文或博客时，要警惕只看漂亮 traversal 图，而忽略 reconstruction 和 likelihood 变差。</p>
          <p><strong>IWAE</strong>，Importance Weighted Autoencoder，用多个 \(z\) 样本构造更紧的似然下界：</p>
          <div class="equation">\[
            \log p_\theta(x)
            \geq
            \mathbb{E}_{z_1,\ldots,z_K\sim q_\phi(z|x)}
            \left[
            \log
            \frac{1}{K}
            \sum_{k=1}^{K}
            \frac{p_\theta(x,z_k)}
            {q_\phi(z_k|x)}
            \right].
          \]</div>
          <p>当 \(K=1\) 时，它退回普通 VAE 的 ELBO。\(K\) 越大，下界通常越紧，但计算更贵。</p>
          <p><strong>Conditional VAE</strong>在生成时加入条件 \(y\)，例如类别标签或文本条件：</p>
          <div class="equation">\[
            p_\theta(x|y)
            =
            \int p_\theta(x|z,y)p(z|y)\,dz.
          \]</div>
          <p>它适合做条件生成，比如给定类别生成图片、给定属性生成样本。核心推导仍然是 ELBO，只是所有分布都多了条件 \(y\)。</p>
        </section>

        <section class="article-section">
          <h2>14.5 读 VAE 论文时应该抓住什么？</h2>
          <p>VAE 系列论文很多，但大多数都可以放回同一个框架里拆解。读论文时可以问五个问题。</p>
          <ol>
            <li><strong>先验 \(p(z)\) 改了吗？</strong>标准 VAE 用 \(\mathcal{N}(0,I)\)。有些工作会使用更复杂先验，让潜空间更贴近聚合后验。</li>
            <li><strong>后验族 \(q_\phi(z|x)\) 改了吗？</strong>标准 VAE 用对角高斯。有些工作用 normalizing flows 增强后验表达能力。</li>
            <li><strong>下界改了吗？</strong>IWAE 改的是下界，用多个重要性样本让 bound 更紧。</li>
            <li><strong>KL 权重改了吗？</strong>β-VAE、capacity annealing 等方法改的是重建和正则之间的权衡。</li>
            <li><strong>decoder likelihood 改了吗？</strong>图像、文本、音频、分子数据的观测分布不同，重建项的含义也不同。</li>
          </ol>
          <p>这样读论文会清楚很多。不要只看模型名字，而要问它到底动了 VAE 框架里的哪一块：先验、后验、下界、decoder、训练策略，还是潜变量解释方式。</p>
        </section>

        <section class="article-section">
          <h2>15. VAE 和 Diffusion 的关系</h2>
          <p>VAE 和 Diffusion 都是生成模型，但思路不同。</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>问题</th><th>VAE</th><th>Diffusion</th></tr>
              </thead>
              <tbody>
                <tr><td>潜变量</td><td>通常是一层或少数层 \(z\)</td><td>整条噪声路径 \(x_1,\ldots,x_T\) 都可看成隐变量</td></tr>
                <tr><td>训练目标</td><td>最大化 ELBO：重建项减 KL</td><td>变分下界可化成逐步去噪或噪声预测目标</td></tr>
                <tr><td>生成方式</td><td>一次采样 \(z\)，一次或少数次解码</td><td>从噪声开始多步迭代采样</td></tr>
                <tr><td>优势</td><td>潜空间清晰、推断快、适合表示学习</td><td>生成质量强，逐步细化能力好</td></tr>
                <tr><td>常见问题</td><td>模糊、posterior collapse、ELBO gap</td><td>采样慢、schedule 和采样器复杂</td></tr>
              </tbody>
            </table>
          </div>
          <p>现代 Latent Diffusion 还会把 VAE 作为压缩模块：先用 VAE 把图像压到 latent space，再在 latent space 里训练 Diffusion。此时 VAE 负责把像素空间压缩成更便宜的连续表示，Diffusion 负责在这个表示空间中建模复杂生成分布。</p>
        </section>

        <section class="article-section">
          <h2>15.5 如何评价一个 VAE？</h2>
          <p>评价 VAE 不能只看一张生成图。至少要分几个层面。</p>
          <p><strong>第一，看 ELBO。</strong>ELBO 是训练目标，也是 \(\log p_\theta(x)\) 的下界。ELBO 越高，通常说明模型对数据的解释越好。但不同 likelihood、不同数据预处理下，ELBO 数值不一定能直接横向比较。</p>
          <p><strong>第二，看重建质量。</strong>输入 \(x\)，经过 encoder 和 decoder 得到重建 \(\hat{x}\)。如果重建很差，说明 \(z\) 或 decoder 没有学到足够信息。但重建太好也不一定代表生成好，因为模型可能只是把训练数据压缩记住了。</p>
          <p><strong>第三，看随机采样。</strong>从 \(p(z)\) 采样再 decode，得到的样本是否自然、多样、覆盖数据分布？这是生成模型最直接的检验。重建好但随机采样差，常常说明潜空间和先验匹配不好。</p>
          <p><strong>第四，看插值和 latent traversal。</strong>插值能检查潜空间是否连续；沿某一维移动能检查潜变量是否学到可解释变化。如果插值中间出现奇怪样本，说明潜空间可能有洞或 decoder 对某些区域不熟悉。</p>
          <p><strong>第五，看 KL 使用情况。</strong>如果 KL 几乎为 0，要警惕 posterior collapse；如果 KL 特别大，可能说明潜空间很不规整，生成时从先验采样会有 mismatch。好的 VAE 通常需要在重建和 KL 之间找到合理平衡。</p>
          <p>所以评价 VAE 时，可以用一句话提醒自己：重建检查 encoder 和 decoder 是否能解释已有数据，采样检查先验和 decoder 是否能生成新数据，KL 检查潜空间是否真的被合理使用。</p>
        </section>

        <section class="article-section">
          <h2>16. 多轮自检：这篇 VAE 是否真的讲通了？</h2>
          <p>最后按初学者视角反复检查。每个问题后面都给出回看位置。如果答不上来，就说明对应环节还没有完全打通。</p>

          <h3>第一轮：对象有没有分清？</h3>
          <p><strong>检查一：你能否解释普通 autoencoder 和 VAE 的根本区别？</strong>普通 autoencoder 通常把 \(x\) 编码成确定性 \(h\)，VAE 把 \(x\) 编码成分布 \(q_\phi(z|x)\)，并且规定可采样先验 \(p(z)\)。卡住时回到第 1 节和第 4 节。</p>
          <p><strong>检查二：你能否写出 VAE 的生成过程？</strong>应该能写出 \(z\sim p(z)\)，\(x\sim p_\theta(x|z)\)，并解释 \(p_\theta(x)=\int p_\theta(x|z)p(z)dz\)。卡住时回到第 2 节。</p>
          <p><strong>检查三：你能否解释 encoder 为什么叫近似后验？</strong>它不是生成模型本身，而是用 \(q_\phi(z|x)\) 近似难算的 \(p_\theta(z|x)\)。卡住时回到第 3 节和第 4 节。</p>

          <h3>第二轮：推导能不能自己复现？</h3>
          <p><strong>检查四：你能否从 KL 分解推出 ELBO？</strong>关键是把 \(p_\theta(z|x)=p_\theta(x|z)p(z)/p_\theta(x)\) 代入 \(D_{\mathrm{KL}}(q_\phi(z|x)\|p_\theta(z|x))\)，然后整理出 \(\log p_\theta(x)=\mathrm{ELBO}+\mathrm{KL}\)。卡住时回到第 5 节。</p>
          <p><strong>检查五：你能否用 Jensen 不等式再推一次 ELBO？</strong>关键是把 \(q_\phi(z|x)\) 乘进去再除出来，把积分写成期望，然后用 \(\log\mathbb{E}[Y]\geq\mathbb{E}[\log Y]\)。卡住时回到第 6 节。</p>
          <p><strong>检查六：你能否推导高斯 KL 闭式？</strong>至少要能在一维情形下写出 \(\log q-\log p\)，再用 \(\mathbb{E}_q[(z-\mu)^2]=\sigma^2\) 和 \(\mathbb{E}_q[z^2]=\mu^2+\sigma^2\)。卡住时回到第 10 节。</p>

          <h3>第三轮：训练和代码能否对应？</h3>
          <p><strong>检查七：你能否解释重参数化技巧为什么必要？</strong>它把 \(z\sim q_\phi(z|x)\) 改写成 \(z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon\)，让随机性来自不依赖参数的 \(\epsilon\)，梯度可以穿过 \(\mu,\sigma\)。卡住时回到第 9 节。</p>
          <p><strong>检查八：你能否解释开山论文里的 SGVB estimator 和 AEVB algorithm？</strong>SGVB 是重参数化后的随机梯度 ELBO 估计器，AEVB 是把这个估计器用于 recognition model 和 generative model 的端到端训练算法。卡住时回到第 9.5 节。</p>
          <p><strong>检查九：你能否解释为什么 pathwise gradient 通常比 score-function estimator 更适合高斯 VAE？</strong>前者让梯度穿过可微采样路径，利用 \(f(z)\) 对 \(z\) 的局部导数；后者更通用但常常方差更大。卡住时回到第 9.6 节。</p>
          <p><strong>检查十：你能否解释 reconstruction loss 为什么有时是 BCE、有时是 MSE？</strong>它取决于 decoder likelihood。Bernoulli likelihood 对应 BCE，固定方差 Gaussian likelihood 对应 MSE。卡住时回到第 8 节。</p>
          <p><strong>检查十一：你能否把代码里的 KL 公式和数学 KL 对上？</strong>代码常用 \(-\frac{1}{2}\sum(1+\mathrm{logvar}-\mu^2-\exp(\mathrm{logvar}))\)，它就是高斯 KL 闭式的 log variance 写法。卡住时回到第 10 节和第 11 节。</p>

          <h3>第四轮：能不能读后续论文？</h3>
          <p><strong>检查十二：你能否解释 β-VAE 改了什么？</strong>它改变的是 KL 正则强度，也就是重建质量、潜空间规整和 disentanglement 之间的权衡。卡住时回到第 14 节。</p>
          <p><strong>检查十三：你能否解释 IWAE 为什么是更紧下界？</strong>它用多个重要性样本估计边缘似然下界，\(K=1\) 时退化为普通 VAE。卡住时回到第 14 节。</p>
          <p><strong>检查十四：你能否说明 VAE 和 Diffusion 的共同点与区别？</strong>共同点是都在最大化或近似最大化数据似然、学习生成分布；区别是 VAE 用潜变量积分和 ELBO，Diffusion 用逐步噪声路径和去噪目标。卡住时回到第 15 节。</p>
          <p><strong>检查十五：你能否评价一个 VAE 是不是学好了？</strong>不能只看重建，也不能只看采样。应该同时看 ELBO、重建、随机采样、插值、latent traversal 和 KL 使用情况。卡住时回到第 12.5 节和第 15.5 节。</p>
          <p>如果这十五个问题能顺畅回答，说明你已经不是在背 VAE 公式，而是真正理解了 VAE 为什么需要变分推断、为什么 ELBO 合理、为什么重参数化能训练、为什么 KL 会塑造潜空间。之后读 Kingma & Welling、Rezende、IWAE、β-VAE 或 Latent Diffusion 相关论文时，就能知道作者是在改先验、改后验族、改下界、改 decoder，还是改潜空间使用方式。</p>
        </section>

        <section class="article-section references">
          <h2>参考资料</h2>
          <ul>
            <li><a href="https://arxiv.org/abs/1312.6114" target="_blank" rel="noreferrer">Kingma and Welling, Auto-Encoding Variational Bayes, 2013</a></li>
            <li><a href="https://arxiv.org/abs/1401.4082" target="_blank" rel="noreferrer">Rezende, Mohamed, Wierstra, Stochastic Backpropagation and Approximate Inference in Deep Generative Models, 2014</a></li>
            <li><a href="https://arxiv.org/abs/1906.02691" target="_blank" rel="noreferrer">Kingma and Welling, An Introduction to Variational Autoencoders, 2019</a></li>
            <li><a href="https://arxiv.org/abs/1509.00519" target="_blank" rel="noreferrer">Burda, Grosse, Salakhutdinov, Importance Weighted Autoencoders, 2015</a></li>
            <li><a href="https://openreview.net/forum?id=Sy2fzU9gl" target="_blank" rel="noreferrer">Higgins et al., beta-VAE, 2017</a></li>
            <li><a href="https://lilianweng.github.io/posts/2018-08-12-vae/" target="_blank" rel="noreferrer">Lilian Weng, From Autoencoder to Beta-VAE</a></li>
          </ul>
        </section>
      `,
    },
    {
      slug: "what-is-generative-model",
      title: "生成模型入门：从概率分布到分布拟合",
      date: "2026-05-22",
      category: "学习笔记",
      summary:
        "一篇真正从零开始的生成模型基础笔记。我们先不讨论 Diffusion、VAE 或 GAN，而是把随机变量、概率分布、总体与样本、似然、最大似然估计、KL 散度和分布拟合这些概念讲清楚。",
      tags: [
        "Generative Model",
        "Probability",
        "Maximum Likelihood",
        "KL Divergence",
        "Machine Learning"
      ],
      cover: "cover-c",
      body: String.raw`
        <p class="lead">在学习 Diffusion、VAE、GAN、Flow Matching 之前，最容易被跳过的问题是：生成模型到底是什么？它为什么总是在说“学习数据分布”？为什么训练目标会变成最大似然、KL 散度或者某种重建损失？这篇文章专门补这个地基。本文参考《深度学习入门5：生成模型》中步骤 1、步骤 2 和步骤 5 的讲法，但不会照搬书中的段落，而是按我们后续学习生成模型需要的逻辑重新组织。</p>

        <section class="article-section">
          <h2>0. 先把一句话说清楚</h2>
          <p>生成模型的核心目标可以先写成一句话：</p>
          <div class="insight-box">生成模型试图学习真实数据背后的概率分布，并利用这个学到的分布生成新的、像真实数据一样的样本。</div>
          <p>这句话里有三个关键词：数据、概率分布、生成。初学者最容易把注意力放在“生成”上，比如生成图片、生成文字、生成音乐。但理论上更根本的是前两个词：数据来自哪里？概率分布是什么？模型为什么可以通过学习分布来生成新样本？</p>
          <p>如果这些问题没想清楚，后面学习 Diffusion 时就会觉得很突然：为什么一张图片要看成高维随机变量？为什么说真实图片服从某个 \(p_{\mathrm{data}}(x)\)？为什么模型要学习 \(p_\theta(x)\)？为什么从高斯噪声开始采样可以得到图片？这些问题的共同根源，都是生成模型的基础概念。</p>
          <p>所以本文先不讲酷炫模型，也先不加可视化。我们只做一件事：把“生成模型是分布建模”这件事讲透。</p>
        </section>

        <section class="article-section">
          <h2>1. 数据不是孤立的点，而是从某个机制中来的</h2>
          <p>假设我们手里有一组身高数据：</p>
          <div class="equation">\[
            171.2,\quad 168.5,\quad 175.0,\quad 180.3,\quad 169.7,\quad \ldots
          \]</div>
          <p>这些数字看起来只是一个表格。但从统计建模的视角看，每个数字都不是凭空出现的，而是从某个总体中抽出来的样本。这个总体可以是“某年龄段男性的身高总体”，也可以是“某学校所有学生的身高总体”。</p>
          <p>总体通常很大，甚至我们永远无法完整观察。我们能拿到的只是有限样本。统计学和机器学习常常做的事情，就是用有限样本反推总体的性质。生成模型也是如此：我们看到有限训练数据，希望学到背后的数据生成机制。</p>

          <p>书里用“总体”和“样本”来解释这个关系。可以把它想成一条链：总体是真实世界中全部对象，采样让我们得到有限样本，参数估计再用有限样本反推总体规律。这个视角非常重要。训练集不是世界本身，训练集只是世界的一小部分观测。模型不能只是记住训练样本，而应该抓住训练样本背后的规律。只有这样，它才能生成训练集中没有的新样本。</p>
          <p>把这个想法换成机器学习语言，就是：</p>
          <div class="equation">\[
            x^{(1)},x^{(2)},\ldots,x^{(N)}
            \sim
            p_{\mathrm{data}}(x).
          \]</div>
          <p>这里 \(x^{(i)}\) 表示第 \(i\) 个观测样本，\(p_{\mathrm{data}}(x)\) 表示真实但未知的数据分布。我们不知道这个分布的完整数学式，只知道训练集像是从它里面抽出来的。</p>
          <p>这一步的心理转变很关键：数据集不是一堆死数据，而是某个未知分布给我们的线索。生成模型要做的，就是利用这些线索还原分布的形状。</p>
        </section>

        <section class="article-section">
          <h2>2. 随机变量和概率分布：生成模型的基本语言</h2>
          <p>为了谈分布，我们先需要“随机变量”。随机变量不是说变量本身很神秘，而是说它的取值由随机机制决定。掷骰子的点数是随机变量，人的身高也可以看成随机变量，一张图片也可以看成随机变量。</p>
          <p>如果随机变量是离散的，比如骰子点数，那么概率分布可以写成每个取值对应的概率：</p>
          <div class="equation">\[
            P(X=1)=\frac{1}{6},\quad
            P(X=2)=\frac{1}{6},\quad
            \ldots,\quad
            P(X=6)=\frac{1}{6}.
          \]</div>
          <p>如果随机变量是连续的，比如身高，单点概率通常没有意义。我们关心的是某个区间内的概率，比如身高在 170 到 175 厘米之间的概率。连续型分布用概率密度函数 \(p(x)\) 描述，区间概率通过积分得到：</p>
          <div class="equation">\[
            P(a\leq X\leq b)=\int_a^b p(x)\,dx.
          \]</div>
          <p>这意味着 \(p(x)\) 本身不是“取值正好等于 \(x\) 的概率”，而是概率密度。真正的概率来自面积，也就是积分。很多初学者第一次看到概率密度会困惑：为什么 \(p(x)\) 可以大于 1？原因就在这里。密度不是概率，面积才是概率。</p>
          <p>生成模型经常处理高维数据。图片可以看成一个向量：</p>
          <div class="equation">\[
            x\in\mathbb{R}^{H\times W\times C}.
          \]</div>
          <p>如果是 \(256\times256\) 的 RGB 图片，那么维度就是 \(256\times256\times3\)。此时 \(p_{\mathrm{data}}(x)\) 是一个极高维空间中的概率分布。这个分布非常复杂：绝大多数随机像素组合都不是自然图片，真正像图片的点只占据空间中很小、很有结构的一部分。</p>
          <p>所以，生成图像并不是“随机拼像素”，而是从一个高度结构化的分布中采样。能否生成像样的图片，取决于模型有没有学到这个分布的结构。</p>
        </section>

        <section class="article-section">
          <h2>3. 什么叫“对分布建模”？</h2>
          <p>真实分布 \(p_{\mathrm{data}}(x)\) 通常未知。我们只能选择一个模型族，用参数控制它的形状。把这个模型分布记作：</p>
          <div class="equation">\[
            p_\theta(x).
          \]</div>
          <p>这里 \(\theta\) 表示模型参数。对于一维正态分布，参数是均值和标准差：</p>
          <div class="equation">\[
            \theta=(\mu,\sigma).
          \]</div>
          <p>此时模型分布是：</p>
          <div class="equation">\[
            p_\theta(x)=
            \frac{1}{\sqrt{2\pi}\sigma}
            \exp\left(
            -\frac{(x-\mu)^2}{2\sigma^2}
            \right).
          \]</div>
          <p>如果我们用正态分布建模身高，那么“建模”就包含两个动作。</p>
          <ol>
            <li>选择模型形式：假设身高可以近似看成正态分布。</li>
            <li>估计模型参数：根据样本决定 \(\mu\) 和 \(\sigma\) 应该是多少。</li>
          </ol>

          <p>书中把这个过程放在身高数据上讲。先观察身高直方图，发现它大致像钟形曲线；于是选择正态分布作为模型；再用样本估计均值和标准差。这个例子虽然简单，但它已经包含生成模型的基本骨架：左边是我们选择的概率模型形式，右边是观测到的样本，中间的参数估计负责把样本信息转化为模型参数。</p>
          <p>更复杂的模型也是同样逻辑。GMM 用多个高斯分布叠加来拟合多峰数据；VAE 用神经网络和潜变量来表达复杂分布；Diffusion 用逐步加噪和反向去噪来学习数据分布。模型形式越来越复杂，但根本问题没变：我们要构造一个 \(p_\theta(x)\)，让它接近 \(p_{\mathrm{data}}(x)\)。</p>
        </section>

        <section class="article-section">
          <h2>4. 生成模型为什么要拟合分布？</h2>
          <p>现在回答最核心的问题：为什么生成模型的目标是拟合分布？</p>
          <p>因为如果模型分布 \(p_\theta(x)\) 足够接近真实分布 \(p_{\mathrm{data}}(x)\)，那么从 \(p_\theta(x)\) 采样出来的样本，就会像真实数据一样。也就是说：</p>
          <div class="equation">\[
            p_\theta(x)\approx p_{\mathrm{data}}(x)
            \quad\Longrightarrow\quad
            x_{\mathrm{new}}\sim p_\theta(x)
            \text{ looks like real data.}
          \]</div>
          <p>这句话把“分布拟合”和“生成新数据”连接起来了。我们不是直接规定每个新样本长什么样，而是先学会什么样的样本概率高，什么样的样本概率低。然后通过采样，就能得到新的合理样本。</p>
          <p>以身高为例，如果模型学到的是均值 172.7、标准差 4.8 的正态分布，那么从这个正态分布采样，得到 173.9、168.2、181.0 这样的数字都很合理。它们不一定出现在原训练集中，但符合身高总体的统计特征。</p>
          <p>以图像为例，真实图片分布会给“有清晰边缘、合理纹理、物体结构正确”的图片较高概率，给随机噪声图片极低概率。如果模型学到了这个分布，那么采样时就更可能得到自然图片，而不是花屏。</p>
          <p>因此，生成模型的“生成能力”来自“分布拟合能力”。拟合分布不是额外的数学包装，而是生成的本质。</p>
        </section>

        <section class="article-section">
          <h2>4.5 训练集如何代表真实分布？</h2>
          <p>这里还有一个细节必须讲清楚：我们说要拟合 \(p_{\mathrm{data}}(x)\)，但训练时根本拿不到这个真实分布的解析式。我们只有训练集。那模型到底是怎么“接触”真实分布的？</p>
          <p>答案是：训练集提供了对真实分布的采样近似。假设样本是独立同分布抽样得到的，那么对于任意函数 \(f(x)\)，真实分布下的期望是：</p>
          <div class="equation">\[
            \mathbb{E}_{p_{\mathrm{data}}}[f(x)]
            =
            \int p_{\mathrm{data}}(x)f(x)\,dx.
          \]</div>
          <p>由于不知道 \(p_{\mathrm{data}}\)，我们用样本平均近似这个期望：</p>
          <div class="equation">\[
            \mathbb{E}_{p_{\mathrm{data}}}[f(x)]
            \approx
            \frac{1}{N}\sum_{i=1}^{N}f(x^{(i)}).
          \]</div>
          <p>这就是蒙特卡洛近似的基本思想：用从分布中抽到的样本，近似分布上的积分。书中在讨论 KL 散度与最大似然关系时，也用到了这个想法。</p>
          <p>如果把训练集本身看成一个经验分布，可以写成：</p>
          <div class="equation">\[
            \hat{p}_{\mathrm{data}}(x)
            =
            \frac{1}{N}
            \sum_{i=1}^{N}
            \delta(x-x^{(i)}).
          \]</div>
          <p>这里的 \(\delta\) 可以理解为“把概率质量放在每个训练样本上的尖峰”。经验分布只在训练样本处有质量，而真实分布应该更平滑、更广泛。训练生成模型时，我们不能只死记这个经验分布，而要用模型族 \(p_\theta(x)\) 学出一个能概括总体规律的分布。</p>
          <p>这也解释了泛化问题。训练集越小，经验分布越粗糙，模型越容易误判总体形状；训练集越大，样本平均越能代表真实分布，参数估计越可靠。但即使训练集很大，模型族如果选得太弱，也仍然学不好。例如用一个单峰正态分布去拟合多种动物图片，模型表达能力显然不够。</p>
          <div class="insight-box">训练集不是目标本身，而是通向真实分布的窗口。生成模型真正想学的是窗口背后的世界。</div>
        </section>

        <section class="article-section">
          <h2>4.6 生成模型和判别模型有什么区别？</h2>
          <p>很多机器学习入门内容会先讲分类模型，比如判断一张图是猫还是狗。这类模型通常关心条件概率：</p>
          <div class="equation">\[
            p(y|x).
          \]</div>
          <p>它回答的问题是：给定输入 \(x\)，标签 \(y\) 是什么？这类模型常被称为判别模型，因为它直接学习如何区分类别边界。</p>
          <p>生成模型关心的问题不同。它更关心数据本身是如何出现的。最朴素的生成模型学习：</p>
          <div class="equation">\[
            p(x).
          \]</div>
          <p>如果是有条件生成，比如给定文字生成图片，则学习：</p>
          <div class="equation">\[
            p(x|c),
          \]</div>
          <p>其中 \(c\) 是条件，例如文本提示、类别标签、草图或其他控制信息。</p>
          <p>判别模型像是在问：“这个样本属于哪一类？”生成模型像是在问：“什么样的样本会自然出现？”前者更关注决策边界，后者更关注数据分布。两者都很重要，但思维方式不同。</p>
          <p>举一个简单例子。假设我们只关心身高和性别。判别模型可能学习 \(p(\text{性别}|\text{身高})\)，即根据身高判断性别；生成模型可能学习 \(p(\text{身高}|\text{性别})\)，即给定性别后生成合理身高，也可能学习联合分布 \(p(\text{身高},\text{性别})\)。</p>
          <p>在现代生成式 AI 里，条件生成非常常见。文本到图像模型学习的是“在给定文本条件下，图片应该服从什么分布”；语言模型学习的是“在给定前文条件下，下一个 token 应该服从什么分布”。这些都是生成建模思想的延伸。</p>
        </section>

        <section class="article-section">
          <h2>5. 似然：用数据反过来评价参数好不好</h2>
          <p>如果我们已经选择了模型族 \(p_\theta(x)\)，接下来要决定参数 \(\theta\)。什么样的参数才算好？一个自然标准是：如果真实观测数据在这个参数下出现的概率很高，那么这个参数就比较合理。</p>
          <p>假设训练集是：</p>
          <div class="equation">\[
            D=\{x^{(1)},x^{(2)},\ldots,x^{(N)}\}.
          \]</div>
          <p>如果每个样本独立来自同一个分布，那么在参数 \(\theta\) 下观测到整个数据集的概率密度是：</p>
          <div class="equation">\[
            p(D;\theta)
            =
            \prod_{i=1}^{N}p_\theta(x^{(i)}).
          \]</div>
          <p>当我们把它看成关于参数 \(\theta\) 的函数时，它叫似然函数：</p>
          <div class="equation">\[
            L(\theta)=p(D;\theta).
          \]</div>
          <p>注意这里有一个视角转换。概率密度 \(p_\theta(x)\) 通常把 \(x\) 当变量，\(\theta\) 固定；似然 \(L(\theta)\) 则把数据 \(D\) 固定，把 \(\theta\) 当变量。我们问的是：哪一组参数最能解释已经发生的这些数据？</p>
          <p>最大似然估计就是选择使似然最大的参数：</p>
          <div class="equation">\[
            \theta^\star=\arg\max_\theta p(D;\theta).
          \]</div>
          <p>实际计算时通常最大化对数似然：</p>
          <div class="equation">\[
            \log p(D;\theta)
            =
            \sum_{i=1}^{N}\log p_\theta(x^{(i)}).
          \]</div>
          <p>为什么取对数？因为乘积会变成求和，计算更稳定，推导也更方便。而且对数函数单调递增，最大化 \(p(D;\theta)\) 和最大化 \(\log p(D;\theta)\) 得到的参数相同。</p>
          <p>从直觉上说，最大似然估计会惩罚那些让真实样本概率很低的参数。如果某个参数把概率质量放在远离训练数据的地方，那么训练集在它下面出现的似然就很低，它自然不是好参数。</p>
        </section>

        <section class="article-section">
          <h2>6. 正态分布例子：最大似然到底估计出了什么？</h2>
          <p>继续看身高例子。我们假设身高服从正态分布：</p>
          <div class="equation">\[
            x\sim\mathcal{N}(\mu,\sigma^2).
          \]</div>
          <p>现在训练集给了我们 \(N\) 个身高值。最大似然估计要找到最能解释这些身高值的 \(\mu\) 和 \(\sigma\)。结果是：</p>
          <div class="equation">\[
            \hat{\mu}=\frac{1}{N}\sum_{i=1}^{N}x^{(i)}.
          \]</div>
          <div class="equation">\[
            \hat{\sigma}^2=
            \frac{1}{N}\sum_{i=1}^{N}
            \left(x^{(i)}-\hat{\mu}\right)^2.
          \]</div>
          <p>也就是说，最大似然给出的正态分布均值就是样本均值，方差就是样本平均平方偏差。这个结论看起来很自然，但它不是凭直觉拍脑袋来的，而是从最大化对数似然推导出来的。</p>
          <p>为了避免这个结果变成“背公式”，我们把推导写完整。正态分布密度是：</p>
          <div class="equation">\[
            p(x;\mu,\sigma^2)
            =
            \frac{1}{\sqrt{2\pi\sigma^2}}
            \exp\left[
            -\frac{(x-\mu)^2}{2\sigma^2}
            \right].
          \]</div>
          <p>由于样本独立同分布，整个数据集的似然是每个样本密度的乘积：</p>
          <div class="equation">\[
            L(\mu,\sigma^2)
            =
            \prod_{i=1}^{N}
            p(x^{(i)};\mu,\sigma^2).
          \]</div>
          <p>取对数后，乘积变成求和：</p>
          <div class="equation">\[
            \ell(\mu,\sigma^2)
            =
            \log L(\mu,\sigma^2)
            =
            -\frac{N}{2}\log(2\pi)
            -\frac{N}{2}\log\sigma^2
            -
            \frac{1}{2\sigma^2}
            \sum_{i=1}^{N}(x^{(i)}-\mu)^2.
          \]</div>
          <p>先对 \(\mu\) 求导。前两项不含 \(\mu\)，只有最后的平方和含 \(\mu\)：</p>
          <div class="equation">\[
            \frac{\partial \ell}{\partial \mu}
            =
            \frac{1}{\sigma^2}
            \sum_{i=1}^{N}(x^{(i)}-\mu).
          \]</div>
          <p>最大值处导数为 0，所以：</p>
          <div class="equation">\[
            \sum_{i=1}^{N}(x^{(i)}-\mu)=0
            \quad\Longrightarrow\quad
            \hat{\mu}=\frac{1}{N}\sum_{i=1}^{N}x^{(i)}.
          \]</div>
          <p>这一步的直觉是：如果均值选得太小，很多样本都在均值右侧，导数会推动 \(\mu\) 变大；如果均值选得太大，导数会推动 \(\mu\) 变小。平衡点正好是样本均值。</p>
          <p>再对 \(\sigma^2\) 求导。为了书写清楚，令 \(s=\sigma^2\)，则：</p>
          <div class="equation">\[
            \ell(\mu,s)
            =
            -\frac{N}{2}\log s
            -
            \frac{1}{2s}
            \sum_{i=1}^{N}(x^{(i)}-\mu)^2
            +\mathrm{const}.
          \]</div>
          <p>对 \(s\) 求导得到：</p>
          <div class="equation">\[
            \frac{\partial \ell}{\partial s}
            =
            -\frac{N}{2s}
            +
            \frac{1}{2s^2}
            \sum_{i=1}^{N}(x^{(i)}-\mu)^2.
          \]</div>
          <p>令导数为 0，并代入 \(\hat{\mu}\)：</p>
          <div class="equation">\[
            -Ns
            +
            \sum_{i=1}^{N}(x^{(i)}-\hat{\mu})^2
            =
            0,
          \]</div>
          <div class="equation">\[
            \hat{\sigma}^2
            =
            \frac{1}{N}
            \sum_{i=1}^{N}(x^{(i)}-\hat{\mu})^2.
          \]</div>
          <p>这里的方差估计分母是 \(N\)，不是统计课里常见的 \(N-1\)。原因是我们现在做的是最大似然估计，它选择让训练数据似然最大的参数；而 \(N-1\) 出现在无偏方差估计里，目标是让估计量在重复抽样意义下无偏。两个公式看起来很像，但优化目标不同。</p>
          <p>这说明一件很重要的事：参数估计不是随便调参数，而是有明确目标函数。我们选择参数，是为了让观测数据在模型下尽可能“合理”。</p>
          <p>当我们把 \(\hat{\mu}\) 和 \(\hat{\sigma}\) 放回正态分布后，就得到了一个完整的生成模型。它可以做两件事：</p>
          <ol>
            <li>生成新数据：从 \(\mathcal{N}(\hat{\mu},\hat{\sigma}^2)\) 采样。</li>
            <li>计算概率：通过密度函数或累积分布函数计算某些事件的概率。</li>
          </ol>
          <p>这正是书中步骤 2.4 强调的两类用途。生成模型不是只能“生成”，它还可以帮助我们进行概率判断。比如估计身高大于某个阈值的概率，本质上也是利用学到的分布。</p>
        </section>

        <section class="article-section">
          <h2>6.5 一个完整的生成建模流程</h2>
          <p>现在把前面的内容压成一个完整流程。以后遇到任何生成模型，都可以先按这几步拆解。</p>
          <p><strong>第一步，确定数据对象。</strong>我们要建模的随机变量是什么？如果是身高，\(x\) 是一个实数；如果是图片，\(x\) 是一个高维张量；如果是文本，\(x\) 是 token 序列。不同数据对象决定了模型需要表达的结构。</p>
          <p><strong>第二步，承认真实分布未知。</strong>我们假设训练样本来自某个 \(p_{\mathrm{data}}(x)\)，但我们不知道它的解析式。这个假设不是为了把问题复杂化，而是为了提醒自己：训练集只是样本，不是总体。</p>
          <p><strong>第三步，选择模型族。</strong>如果数据简单，可以选择正态分布；如果数据多峰，可以选择 GMM；如果数据是图像，就需要神经网络构造复杂分布。这个选择决定了模型的表达能力，也决定了训练和采样方式。</p>
          <p><strong>第四步，定义训练目标。</strong>最经典的是最大似然：</p>
          <div class="equation">\[
            \max_\theta
            \sum_{i=1}^{N}
            \log p_\theta(x^{(i)}).
          \]</div>
          <p>如果精确似然不好算，就可能使用变分下界、对抗目标、score matching、denoising objective 等替代目标。替代目标看起来不同，但通常仍然服务于同一个目的：让模型分布接近真实数据分布。</p>
          <p><strong>第五步，优化参数。</strong>对于一维正态分布，最大似然有解析解；对于神经网络生成模型，通常用梯度下降。优化算法本身不是生成模型的本质，它只是找到好参数的工具。</p>
          <p><strong>第六步，使用模型。</strong>训练完成后，我们可以从模型中采样，也可以用模型做概率评价、异常检测、缺失数据补全或条件生成。模型的价值来自它学到的分布，而不仅仅是生成一张好看的图。</p>
          <p>这六步看起来朴素，但非常通用。后续学 Diffusion 时，也可以这样对应：数据对象是图片或 latent；真实分布是图片数据分布；模型族是由 U-Net/Transformer 参数化的反向过程或 score；训练目标是噪声预测或变分目标；优化用梯度下降；使用时从噪声采样并逐步生成。</p>
        </section>

        <section class="article-section">
          <h2>7. 拟合分布和最大似然为什么是一回事？</h2>
          <p>上面我们说最大似然会选出能解释数据的参数。但它和“拟合真实分布”之间是什么关系？这里需要引入 KL 散度。</p>
          <p>KL 散度可以衡量两个分布之间的差异。对于真实分布 \(p_{\mathrm{data}}(x)\) 和模型分布 \(p_\theta(x)\)，一个常见方向是：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}
            \left(
            p_{\mathrm{data}}
            \,\|\,p_\theta
            \right)
            =
            \int
            p_{\mathrm{data}}(x)
            \log
            \frac{p_{\mathrm{data}}(x)}{p_\theta(x)}
            dx.
          \]</div>
          <p>KL 散度越小，说明模型分布越接近真实分布。当两个分布完全相同时，KL 散度为 0。</p>

          <p>如果随机变量是离散的，KL 散度写成求和形式：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}(p\|q)
            =
            \sum_x p(x)\log\frac{p(x)}{q(x)}.
          \]</div>
          <p>这个式子可以逐项理解。每个 \(x\) 都由 \(p(x)\) 加权，这表示我们主要关心真实分布 \(p\) 经常出现的位置；后面的 \(\log\frac{p(x)}{q(x)}\) 则比较真实概率和模型概率。如果某个位置真实概率很高，但模型概率很低，那么这一项会变大，KL 散度会惩罚模型。</p>

          <p>用书中的硬币例子可以这样理解。真实硬币正面概率是 0.7，反面概率是 0.3。如果模型 \(q_1\) 认为正反面都是 0.5，它没有完全正确，但还算接近，所以 KL 较小。如果模型 \(q_2\) 认为正面只有 0.2、反面有 0.8，它和真实分布方向几乎相反，所以 KL 较大。如果模型 \(q_3\) 完全等于真实分布，那么 KL 为 0。</p>
          <p>KL 散度有一个容易误解的地方：它不是普通距离。普通距离通常满足对称性，比如从 A 到 B 的距离等于从 B 到 A 的距离；但 KL 散度一般不对称：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}(p\|q)
            \neq
            D_{\mathrm{KL}}(q\|p).
          \]</div>
          <p>为什么会不对称？因为 \(D_{\mathrm{KL}}(p\|q)\) 是用 \(p(x)\) 加权的，它更关心真实分布 \(p\) 常出现的位置。如果在某个真实高概率区域，模型 \(q\) 给了极低概率，惩罚会非常重。反过来，\(D_{\mathrm{KL}}(q\|p)\) 又会用 \(q(x)\) 加权，关注点就变了。</p>
          <p>一个极端情况最能说明问题：如果某个位置 \(p(x)>0\)，但 \(q(x)=0\)，那么 \(\log\frac{p(x)}{q(x)}\) 会发散，KL 变成无穷大。直觉上，这是因为真实世界明明会出现某类样本，但模型说它绝不可能出现，这个错误非常严重。</p>

          <p>我们想让 \(p_\theta(x)\) 接近 \(p_{\mathrm{data}}(x)\)，也就是最小化：</p>
          <div class="equation">\[
            \arg\min_\theta
            D_{\mathrm{KL}}
            \left(
            p_{\mathrm{data}}
            \,\|\,p_\theta
            \right).
          \]</div>
          <p>展开 KL 散度：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}
            =
            \int p_{\mathrm{data}}(x)\log p_{\mathrm{data}}(x)\,dx
            -
            \int p_{\mathrm{data}}(x)\log p_\theta(x)\,dx.
          \]</div>
          <p>第一项只和真实分布有关，不包含 \(\theta\)，所以优化参数时可以看成常数。真正影响 \(\theta\) 的是第二项：</p>
          <div class="equation">\[
            \arg\min_\theta D_{\mathrm{KL}}
            =
            \arg\max_\theta
            \int p_{\mathrm{data}}(x)\log p_\theta(x)\,dx.
          \]</div>
          <p>但我们不知道 \(p_{\mathrm{data}}(x)\) 的完整形式，只能用训练样本近似这个期望。于是得到：</p>
          <div class="equation">\[
            \int p_{\mathrm{data}}(x)\log p_\theta(x)\,dx
            \approx
            \frac{1}{N}\sum_{i=1}^{N}
            \log p_\theta(x^{(i)}).
          \]</div>
          <p>右边就是平均对数似然。于是我们得到一个非常重要的结论：</p>
          <div class="insight-box">在样本足够多时，最大化训练数据的对数似然，可以理解为最小化真实数据分布和模型分布之间的 KL 散度。</div>
          <p>这就是“为什么最大似然是在拟合分布”的理论连接。书中步骤 5.1.3 也强调了这一点：最大似然估计可以从最小化 KL 散度的角度得到。</p>
          <p>所以，当我们训练生成模型时，最大似然不是单纯让训练样本概率变大，它背后的目标是让模型分布更接近真实分布。训练集只是我们接触真实分布的方式。</p>
        </section>

        <section class="article-section">
          <h2>7.5 负对数似然、交叉熵和训练 loss 是什么关系？</h2>
          <p>如果你看深度学习代码，常常不会直接看到“最大似然”四个字，而是看到 loss。最常见的写法是最小化负对数似然：</p>
          <div class="equation">\[
            \mathcal{L}_{\mathrm{NLL}}(\theta)
            =
            -\frac{1}{N}
            \sum_{i=1}^{N}
            \log p_\theta(x^{(i)}).
          \]</div>
          <p>这和最大似然完全等价。最大化平均对数似然，就是最小化它的相反数。工程里习惯“最小化 loss”，所以把目标写成负号形式。</p>
          <p>从分布角度看，真实分布到模型分布的 KL 可以写成：</p>
          <div class="equation">\[
            D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)
            =
            \mathbb{E}_{p_{\mathrm{data}}}
            [\log p_{\mathrm{data}}(x)]
            -
            \mathbb{E}_{p_{\mathrm{data}}}
            [\log p_\theta(x)].
          \]</div>
          <p>第一项只和真实分布有关，叫负熵的一部分；训练时它不含 \(\theta\)，不能通过调模型改变。第二项才和模型有关。把符号换一下，就得到交叉熵：</p>
          <div class="equation">\[
            H(p_{\mathrm{data}},p_\theta)
            =
            -
            \mathbb{E}_{p_{\mathrm{data}}}
            [\log p_\theta(x)].
          \]</div>
          <p>因此，最小化负对数似然、最小化交叉熵、最小化 \(D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)\)，在这里本质上是在做同一件事：让模型给真实数据常出现的区域更高概率。</p>
          <p>这个连接非常重要，因为后面很多生成模型会把目标包装成不同形式。VAE 里会出现 ELBO，Diffusion 里会出现变分下界和噪声 MSE，语言模型里会出现 token-level cross entropy。表面上 loss 不同，底层问题仍然是：模型分布是否更接近真实数据分布。</p>
        </section>

        <section class="article-section">
          <h2>8. 生成模型的两种能力：采样和评价</h2>
          <p>一个明确的概率生成模型通常有两种能力。</p>
          <p><strong>第一，采样。</strong>如果我们能从 \(p_\theta(x)\) 中抽样，就能得到新数据：</p>
          <div class="equation">\[
            x_{\mathrm{new}}\sim p_\theta(x).
          \]</div>
          <p>身高模型中，采样就是从一个正态分布随机生成新身高。图像模型中，采样就是生成新图片。Diffusion 中，从高斯噪声开始逐步去噪，本质上也是一种采样过程。</p>

          <p>书中用真实身高数据和生成身高数据的直方图来说明这一点：生成模型不是要求生成样本逐个等于训练样本，而是要求生成样本整体形成的分布和真实数据分布大体重合。换句话说，判断生成模型好不好，不能只看某一个样本，而要看模型采样出来的一批样本是否具有真实数据的统计特征。</p>
          <p><strong>第二，概率评价。</strong>如果我们能计算 \(p_\theta(x)\) 或相关概率，就能判断某个样本在模型下有多合理。例如，身高 172 厘米在模型下密度较高，身高 230 厘米密度极低。对于图像，理想情况下自然图片应该有高概率，随机噪声应该有低概率。</p>
          <p>不同生成模型对这两种能力的支持程度不同。有些模型很容易采样，但不容易计算精确似然，比如 GAN。有些模型可以计算或近似似然，比如 VAE、Flow、Autoregressive Model。有些模型训练和采样路径比较特殊，比如 Diffusion。</p>
          <p>这也解释了为什么生成模型有很多类别。它们都想学习数据分布，但选择了不同的表达方式、训练目标和采样方法。</p>
        </section>

        <section class="article-section">
          <h2>9. 从一维身高到高维图像，困难在哪里？</h2>
          <p>一维身高数据可以用正态分布近似，这是因为它结构简单。一个均值和一个标准差就能描述主要形状。但图像分布远远复杂得多。</p>
          <p>首先，图像是高维对象。一个样本不是一个数字，而是几十万维的向量。高维空间里，大部分点都没有意义。随机采样像素几乎一定得到噪声图。</p>
          <p>其次，图像分布有复杂依赖。相邻像素有关联，局部纹理和整体语义有关联，物体形状和背景也有关联。简单正态分布无法表达这些关系。</p>
          <p>再次，图像分布通常是多峰的。猫、狗、车、人脸、风景都属于自然图片，但它们集中在高维空间中的不同区域。一个单峰高斯不可能同时很好地描述这些复杂类别。</p>
          <p>因此，现代生成模型需要更强的模型族。神经网络的作用，就是提供足够强的函数表达能力。它可以把简单噪声映射成复杂样本，也可以预测去噪方向，还可以参数化条件分布。</p>
          <p>但是不要被复杂网络吓住。无论模型多复杂，底层问题仍然是：</p>
          <div class="equation">\[
            \text{Find } p_\theta(x)
            \text{ such that }
            p_\theta(x)\approx p_{\mathrm{data}}(x).
          \]</div>
          <p>这就是生成模型的共同主线。</p>
        </section>

        <section class="article-section">
          <h2>10. 几类生成模型在同一张地图上的位置</h2>
          <p>现在我们可以把常见生成模型放到同一张概念地图里。它们不是互不相干的名词，而是围绕同一个分布建模目标给出的不同方案。</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>模型类型</th><th>核心想法</th><th>和分布拟合的关系</th></tr>
              </thead>
              <tbody>
                <tr><td>正态分布模型</td><td>用均值和方差描述一维或多维数据</td><td>最简单的显式概率模型，可以直接最大似然估计。</td></tr>
                <tr><td>高斯混合模型</td><td>用多个高斯分布叠加描述多峰数据</td><td>比单个高斯更灵活，引入潜变量表示样本来自哪个成分。</td></tr>
                <tr><td>VAE</td><td>用潜变量和神经网络生成数据</td><td>通过 ELBO 近似最大化似然，学习潜变量分布和解码器。</td></tr>
                <tr><td>GAN</td><td>生成器和判别器对抗训练</td><td>不直接写出似然，而是通过对抗方式让生成分布接近真实分布。</td></tr>
                <tr><td>Flow</td><td>用可逆变换把简单分布变成复杂分布</td><td>可以精确计算密度，直接进行最大似然训练。</td></tr>
                <tr><td>Diffusion</td><td>先把数据逐步加噪，再学习反向去噪</td><td>通过噪声条件分布、变分目标或 score 学习数据分布。</td></tr>
              </tbody>
            </table>
          </div>
          <p>如果只记模型名字，很容易迷路。更好的学习方式是每遇到一个模型，就问三个问题：</p>
          <ol>
            <li>它如何表达 \(p_\theta(x)\)？</li>
            <li>它用什么目标让 \(p_\theta(x)\) 接近 \(p_{\mathrm{data}}(x)\)？</li>
            <li>训练完成后，它如何从 \(p_\theta(x)\) 中采样？</li>
          </ol>
          <p>这三个问题能把大多数生成模型串起来。</p>

          <figure class="source-figure">
            <img src="https://upload.wikimedia.org/wikipedia/commons/3/34/Three-generative-models.png" alt="自回归模型、VAE 和 GAN 的生成模型示意图" loading="lazy" />
            <figcaption>参考图：三类经典生成模型的结构对比。读图时不要只看网络形状，而要问它们分别怎样表达分布、怎样训练、怎样采样。图片来源：Wikimedia Commons, Three-generative-models.png。</figcaption>
          </figure>
        </section>

        <section class="article-section">
          <h2>10.5 从这张地图走向开山论文</h2>
          <p>如果只停留在“VAE、GAN、Flow、Diffusion 都是生成模型”这一层，还是会觉得这些名词有点散。更好的办法是把每类模型和它最早解决的问题对上。下面这张表不是为了背论文名，而是为了建立读论文时的坐标系：这篇论文究竟改变了 \(p_\theta(x)\) 的表达方式、训练目标，还是采样方式？</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>路线</th><th>代表性开端</th><th>它解决的核心问题</th><th>回到本文应该看哪里</th></tr>
              </thead>
              <tbody>
                <tr><td>潜变量模型 / VAE</td><td>Kingma & Welling, Auto-Encoding Variational Bayes；Rezende et al., Stochastic Backpropagation</td><td>边缘似然含有难积分的潜变量，怎么办？答案是用变分后验、ELBO 和重参数化，让神经网络潜变量生成模型可以端到端训练。</td><td>第 3 节、第 7 节、第 10 节。</td></tr>
                <tr><td>GAN</td><td>Goodfellow et al., Generative Adversarial Nets</td><td>如果不显式写出可计算似然，能不能仍然让生成分布接近真实分布？GAN 用判别器给生成器提供分布差异信号。</td><td>第 4 节、第 8 节、第 10 节。</td></tr>
                <tr><td>Normalizing Flow</td><td>Dinh et al., NICE；后续 Real NVP、Glow</td><td>能不能既容易采样，又能精确算密度？Flow 用可逆变换和变量替换公式，把简单分布变成复杂分布。</td><td>第 5 节、第 6 节、第 10 节。</td></tr>
                <tr><td>Diffusion / Score</td><td>Ho et al., Denoising Diffusion Probabilistic Models；Song et al., Score-Based Generative Modeling through SDEs</td><td>复杂数据分布难以直接采样，能不能通过一条从数据到噪声、再从噪声回到数据的路径来学习？Diffusion 把分布拟合拆成许多噪声等级上的局部去噪问题。</td><td>第 2 节、第 4 节、第 7 节、第 10 节。</td></tr>
                <tr><td>Flow Matching</td><td>Lipman et al., Flow Matching for Generative Modeling</td><td>能不能直接学习把简单分布运输到数据分布的速度场？Flow Matching 用连续性方程和 probability path，把生成看成概率质量的连续流动。</td><td>第 4 节、第 8 节、第 10 节。</td></tr>
              </tbody>
            </table>
          </div>
          <p>这样看，现代生成模型不是一堆孤立技巧，而是一组围绕同一目标的不同选择：有的选择显式密度，有的选择隐式分布匹配；有的引入潜变量，有的引入可逆变换，有的引入噪声路径或速度场。读论文时只要抓住“它在怎样让 \(p_\theta\) 接近 \(p_{\mathrm{data}}\)”这个问题，就不会被符号淹没。</p>
        </section>

        <section class="article-section">
          <h2>11. 常见误解</h2>
          <h3>误解一：生成模型就是会画图的模型</h3>
          <p>画图只是生成模型的一种应用。生成模型可以生成图像、音频、文本、分子结构，也可以生成一维身高数据。只要目标是学习数据分布并从中产生新样本，就属于生成建模的范畴。</p>

          <h3>误解二：生成模型只需要记住训练集</h3>
          <p>如果模型只是记住训练样本，它无法真正生成新数据。好的生成模型要学习总体规律，而不是复制样本。它生成的新样本应该没有出现在训练集中，但仍符合训练数据背后的分布。</p>

          <h3>误解三：概率密度越大就等于概率越大</h3>
          <p>连续变量中，密度不是单点概率。真正的概率来自区间上的积分。这个区别在理解似然和概率密度时非常重要。</p>

          <h3>误解四：最大似然只是一个工程损失</h3>
          <p>最大似然背后有明确的分布拟合意义。它可以看成用训练样本近似真实分布下的期望，从而最小化真实分布到模型分布的 KL 散度。</p>

          <h3>误解五：深度生成模型和简单概率模型完全无关</h3>
          <p>深度生成模型只是把 \(p_\theta(x)\) 做得更复杂、更有表达力。它们仍然继承了概率建模的基本问题：分布是什么，参数是什么，训练目标是什么，采样过程是什么。</p>
        </section>

        <section class="article-section">
          <h2>12. 读完本文后应该掌握什么？</h2>
          <p>如果这篇文章读懂了，你应该能用自己的话回答下面几个问题。</p>
          <ol>
            <li>为什么训练集可以看成从真实数据分布 \(p_{\mathrm{data}}(x)\) 中采样得到？</li>
            <li>为什么生成模型不是直接背数据，而是学习一个模型分布 \(p_\theta(x)\)？</li>
            <li>为什么 \(p_\theta(x)\approx p_{\mathrm{data}}(x)\) 后，从 \(p_\theta(x)\) 采样就能生成像真实数据的新样本？</li>
            <li>似然 \(p(D;\theta)\) 和概率密度 \(p_\theta(x)\) 的视角有什么区别？</li>
            <li>正态分布最大似然为什么会推出样本均值和样本方差？</li>
            <li>为什么最大似然估计可以看成分布拟合？</li>
            <li>负对数似然、交叉熵和 KL 散度之间是什么关系？</li>
            <li>为什么从身高这种一维生成模型到图像生成模型，困难主要来自高维和复杂依赖？</li>
            <li>为什么 VAE、GAN、Flow、Diffusion 和 Flow Matching 都可以放在同一张“分布拟合”地图上？</li>
          </ol>
          <p>这些问题就是后续学习 Diffusion 的地基。Diffusion 的前向加噪、反向去噪、score、ELBO、Flow Matching 等概念，本质上都建立在“学习数据分布”这件事上。</p>
        </section>

        <section class="article-section">
          <h2>13. 多轮自检：初学者最容易卡在哪里？</h2>
          <p>写到这里，我会用一个完全初学者的视角反复检查：有没有某句话听起来懂了，但其实还不知道为什么？有没有某个公式可以背下来，但不知道它解决了什么问题？下面不是为了凑问答，而是给读者一个回看索引。每个问题后面都标出应该回到哪里找答案。</p>

          <h3>第一轮：定义有没有真正落地？</h3>
          <p><strong>问题一：为什么不能直接说“生成模型就是学习训练集”？</strong>因为训练集只是有限样本。一个模型如果只记住训练集，它最多能复读旧样本，不能理解总体规律。生成模型真正想学的是训练集背后的数据分布。对于身高数据，训练集中可能没有 173.9 厘米这个值，但如果总体分布支持这个身高，模型就应该能生成它。对于图片也是一样，模型生成的新图片不应该只是训练图像的复制品，而应该是符合真实图片分布的新样本。卡住时回到第 1 节和第 4.5 节。</p>
          <p><strong>问题二：为什么一定要引入概率分布？不能只训练一个函数吗？</strong>如果只训练一个确定性函数，输入相同就输出相同，很难描述“可能性”。生成问题天然带有不确定性：同一句提示词可以对应许多合理图片，同一个类别可以对应许多不同样本。概率分布提供了一种语言，告诉我们哪些样本更可能出现，哪些样本不太可能出现。没有分布，就很难严格讨论“像真实数据”这件事。卡住时回到第 2 节和第 4 节。</p>
          <p><strong>问题三：什么叫“从模型里采样”？</strong>采样不是让模型复制某个训练样本，而是按照模型分布的概率规律随机产生一个新样本。身高模型里，采样就是从正态分布抽一个身高；图像生成里，采样就是从学到的高维图像分布里抽一个图像。卡住时回到第 8 节。</p>

          <h3>第二轮：训练目标有没有讲通？</h3>
          <p><strong>问题四：似然里的概率密度很小，为什么还要最大化它？</strong>在连续高维空间里，单个样本的密度值可能非常小，这是正常的。最大似然不是关心绝对数值好不好看，而是比较不同参数下训练数据的相对合理性。参数 A 下训练数据密度更高，说明 A 比参数 B 更能解释这些观测。实际计算时我们用对数似然，把很多小数的乘积变成求和，也避免数值下溢。卡住时回到第 5 节和第 6 节。</p>
          <p><strong>问题五：正态分布最大似然的样本均值和样本方差从哪里来？</strong>它们不是经验猜测，而是对正态分布对数似然分别对 \(\mu\) 和 \(\sigma^2\) 求导、令导数为 0 得到的。卡住时回到第 6 节，把每一步求导重新写一遍。</p>
          <p><strong>问题六：KL 散度到底在惩罚什么？</strong>对于 \(D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_\theta)\)，它重点惩罚模型在真实数据高概率区域给出低概率。也就是说，真实世界常见的东西，模型不能说它罕见；真实世界会出现的模式，模型不能完全漏掉。这和生成模型的目标一致：模型必须覆盖真实数据的重要区域。卡住时回到第 7 节，尤其是硬币例子和 KL 展开。</p>
          <p><strong>问题七：最大似然、负对数似然、交叉熵和 KL 散度是不是不同目标？</strong>在本文语境里，它们是同一件事的不同写法。最大似然从样本角度出发，负对数似然是为了写成最小化 loss，交叉熵是分布期望形式，KL 则再减去一个与模型无关的真实分布熵项。卡住时回到第 7 节和第 7.5 节。</p>

          <h3>第三轮：能不能迁移到现代生成模型？</h3>
          <p><strong>问题八：为什么一个简单正态分布也算生成模型？</strong>因为它满足生成模型的定义：它对数据分布进行建模，并且可以从模型分布中采样新数据。不要把“生成模型”只理解成大型神经网络。正态分布模型、高斯混合模型、VAE、Flow、Diffusion 都在同一条线上，只是表达能力和训练方式不同。卡住时回到第 3 节、第 6 节和第 10 节。</p>
          <p><strong>问题九：如果模型分布接近真实分布，为什么单个生成样本仍可能不好？</strong>分布相近说的是整体统计规律，而不是每一次采样都完美。一个模型可能总体上覆盖了真实数据的主要模式，但某些采样点仍然落在低质量区域。评价生成模型时，不能只看一张图，也不能只看一个数字，而要同时看样本质量、多样性、分布覆盖和是否过拟合。卡住时回到第 8 节和第 9 节。</p>
          <p><strong>问题十：这篇文章和下一篇 Diffusion 有什么关系？</strong>Diffusion 也是生成模型，所以它仍然要学习 \(p_{\mathrm{data}}(x)\)。区别在于它不直接写出一个简单的 \(p_\theta(x)\)，而是构造一条从数据到噪声、再从噪声回到数据的路径。理解了本文的分布、似然、KL、采样之后，再看 Diffusion 的前向加噪、反向去噪、ELBO 和 score，就会知道它们不是孤立技巧，而是在服务同一个目标：让模型分布接近真实数据分布。卡住时回到第 10 节，然后进入下一篇。</p>

          <h3>第四轮：能不能接上开山论文？</h3>
          <p><strong>问题十一：为什么读 VAE、GAN、Flow、Diffusion 的开山论文之前，必须先懂“分布拟合”？</strong>因为这些论文表面上改的是网络结构或训练技巧，背后都在处理同一个问题：怎样表示、训练并采样一个接近真实数据分布的 \(p_\theta(x)\)。如果这个主线不清楚，ELBO、对抗损失、变量替换公式、去噪 MSE、速度场回归都会变成孤立公式。卡住时回到第 4 节、第 7 节和第 10.5 节。</p>
          <p><strong>问题十二：为什么有些模型能算似然，有些模型不能精确算似然，却仍然都叫生成模型？</strong>生成模型的定义不是“必须能精确算 \(p_\theta(x)\)”，而是“能够刻画数据分布并从中生成样本”。Flow 和 autoregressive model 通常能精确或近似计算似然；VAE 用 ELBO 近似；GAN 常用隐式分布匹配；Diffusion 通过多步去噪和变分目标学习采样路径。卡住时回到第 8 节、第 10 节和第 10.5 节。</p>
        </section>

        <section class="article-section references">
          <h2>参考资料</h2>
          <ul>
            <li>斋藤康毅 著，郑明智 译，《深度学习入门5：生成模型》，人民邮电出版社，2025。本文主要参考步骤 1“正态分布”、步骤 2“最大似然估计”和步骤 5.1“KL 散度”；文中的总体/样本、参数估计、真实与生成分布、KL 硬币例子均为根据书中相关图意改写成文字说明。</li>
            <li><a href="https://arxiv.org/abs/1312.6114" target="_blank" rel="noreferrer">Kingma and Welling, Auto-Encoding Variational Bayes, 2013</a></li>
            <li><a href="https://arxiv.org/abs/1401.4082" target="_blank" rel="noreferrer">Rezende, Mohamed, Wierstra, Stochastic Backpropagation and Approximate Inference in Deep Generative Models, 2014</a></li>
            <li><a href="https://arxiv.org/abs/1406.2661" target="_blank" rel="noreferrer">Goodfellow et al., Generative Adversarial Nets, 2014</a></li>
            <li><a href="https://arxiv.org/abs/1410.8516" target="_blank" rel="noreferrer">Dinh, Krueger, Bengio, NICE: Non-linear Independent Components Estimation, 2014</a></li>
            <li><a href="https://arxiv.org/abs/2006.11239" target="_blank" rel="noreferrer">Ho, Jain, Abbeel, Denoising Diffusion Probabilistic Models, 2020</a></li>
            <li><a href="https://arxiv.org/abs/2011.13456" target="_blank" rel="noreferrer">Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, 2020</a></li>
            <li><a href="https://arxiv.org/abs/2210.02747" target="_blank" rel="noreferrer">Lipman et al., Flow Matching for Generative Modeling, 2022</a></li>
            <li><a href="https://lilianweng.github.io/posts/2018-10-13-flow-models/" target="_blank" rel="noreferrer">Lilian Weng, Flow-based Deep Generative Models</a></li>
            <li><a href="https://commons.wikimedia.org/wiki/File:Three-generative-models.png" target="_blank" rel="noreferrer">Wikimedia Commons, Three-generative-models.png</a></li>
          </ul>
        </section>
      `,
    },
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
            <img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png" alt="生成模型类型对比图" loading="lazy" />
            <figcaption>参考图：GAN、VAE、Flow-based model 和 Diffusion model 的结构对比。Diffusion 可以看成从数据到噪声、再从噪声回到数据的建模路线。图片来源：Lilian Weng, What are Diffusion Models?</figcaption>
          </figure>

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
            <img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-beta.png" alt="线性和余弦噪声日程对比" loading="lazy" />
            <figcaption>参考图：不同噪声 schedule 下 \(\bar{\alpha}_t\) 的衰减方式。它帮助理解不同时间步中原始信号还保留多少。图片来源：Lilian Weng, What are Diffusion Models?</figcaption>
          </figure>

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
            <img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png" alt="DDPM 前向扩散和反向去噪示意图" loading="lazy" />
            <figcaption>参考图：DDPM 的前向扩散链和反向去噪链。读图时重点看方向：前向 \(q\) 是人为设计的加噪，反向 \(p_\theta\) 才是模型学习的生成过程。图片来源：Lilian Weng, What are Diffusion Models?</figcaption>
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

          <figure class="source-figure">
            <img src="https://mlg.eng.cam.ac.uk/blog/assets/images/flow-matching/g2g-cond-paths-one-color.png" alt="Flow Matching 条件路径示意图" loading="lazy" />
            <figcaption>参考图：Flow Matching 中从源分布到目标分布的条件路径。它帮助理解为什么训练时可以有条件速度监督，而生成时模型学到的是边缘速度场。图片来源：Cambridge MLG Blog, Flow Matching Guide and Code。</figcaption>
          </figure>

          <figure class="source-figure">
            <img src="https://diffusionflow.github.io/assets/img/2025-04-28-distill-example/particle_movement.gif" alt="DiffusionFlow 粒子沿速度场移动示意动图" loading="lazy" />
            <figcaption>参考动图：把样本看成粒子沿速度场从简单分布移动到数据分布，有助于把 Flow Matching 的 ODE 视角和 Diffusion 的分布运输视角连起来。图片来源：DiffusionFlow。</figcaption>
          </figure>

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
          <h2>21. 多轮自检：你是否真的能继续读论文了？</h2>
          <p>最后不只做一次检查，而是从三个层次反复问自己：定义是否清楚，推导是否能复现，读论文时是否知道作者在改什么。每个问题后面都给出回看位置。如果某个问题答不上来，不是失败，而是说明需要回到对应章节重新推一遍。</p>

          <h3>第一轮：基本对象有没有混淆？</h3>
          <p><strong>检查一：你能否解释 \(x_0,x_t,x_T,\epsilon,\epsilon_\theta\) 分别是什么？</strong>\(x_0\) 是干净数据，\(x_t\) 是第 \(t\) 个噪声等级的样本，\(x_T\) 应该接近标准高斯，\(\epsilon\) 是真实采样噪声，\(\epsilon_\theta\) 是网络预测噪声。卡住时回到第 0 节符号表和第 3 节。</p>
          <p><strong>检查二：你能否解释为什么 \(q\) 不是模型、\(p_\theta\) 才是模型？</strong>\(q\) 是人为规定的前向加噪过程，它不学习；\(p_\theta\) 是反向生成过程，需要神经网络学习。卡住时回到第 5 节。</p>
          <p><strong>检查三：你能否解释为什么 \(q(x_t|x_0)\) 可以一步采样？</strong>回答时应该说出独立高斯线性组合仍是高斯，信号系数累计成 \(\sqrt{\bar{\alpha}_t}\)，噪声方差累计成 \(1-\bar{\alpha}_t\)。如果只能背公式，还不算真正理解。卡住时回到第 2.5 节和第 4 节。</p>

          <h3>第二轮：核心推导能不能自己走完？</h3>
          <p><strong>检查四：你能否解释为什么 \(q(x_{t-1}|x_t,x_0)\) 是高斯？</strong>回答时应该说出贝叶斯公式、Markov 性、两个高斯密度相乘、二次型配方、精度相加。这里是读 DDPM 论文最关键的门槛。卡住时回到第 6 节。</p>
          <p><strong>检查五：你能否解释为什么预测噪声可以训练生成模型？</strong>回答时不能只说“DDPM 就是这么做的”，而要说清楚：预测噪声可以反推出 \(\hat{x}_0\)，进而参数化反向均值；反向高斯 KL 在方差固定时变成均值 MSE；均值 MSE 又可以写成噪声 MSE。卡住时回到第 7 节和第 9 节。</p>
          <p><strong>检查六：你能否把 score 和噪声预测联系起来？</strong>应该能写出：</p>
          <div class="equation">\[
            \nabla_{x_t}\log q(x_t|x_0)
            =
            -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
          \]</div>
          <p>并且能解释这只是条件 score，真正生成需要边缘 score，而 denoising score matching 正是用条件监督学习边缘方向。卡住时回到第 8 节。</p>

          <h3>第三轮：能不能带着框架读新论文？</h3>
          <p><strong>检查七：你能否解释 SDE、ODE、Flow Matching 的共同点？</strong>它们都在描述概率分布如何从噪声端移动到数据端。SDE 用随机过程和 score 修正反向漂移，probability flow ODE 用确定性速度保持同样边缘分布，Flow Matching 直接回归满足连续性方程的速度场。卡住时回到第 11 节和第 12 节。</p>
          <p><strong>检查八：你能否看到一篇新论文时，立刻定位它改变的是哪一层？</strong>有些论文改变噪声 schedule，有些改变网络输出参数化，有些改变采样器，有些把像素空间换成 latent space，有些把 score 语言换成 flow 语言。只要能把创新点放回“路径、目标、参数化、采样”这四个位置，就不会被新名词牵着走。卡住时回到第 14 节、第 15 节和第 17 节。</p>
          <p><strong>检查九：你能否把公式翻译成口头解释？</strong>比如 \(\bar{\alpha}_t\) 不是一个神秘符号，而是原始信号保留比例；\(\tilde{\mu}_t\) 不是凭空出现的均值，而是由 \(x_t\) 和 \(x_0\) 两个信息源按可靠性加权；score 不是抽象梯度，而是往高概率区域移动的方向。能做到这一点，说明公式已经变成理解，而不是记忆负担。卡住时回到第 4.5 节、第 6 节和第 8 节。</p>
          <p><strong>检查十：你能否解释为什么采样慢、为什么可以加速？</strong>慢是因为模型沿着很多离散噪声等级逐步移动，每一步只做局部修正；加速则来自更好的时间步选择、更稳定的 ODE/SDE 求解器、或者更直接的路径学习。这样看 DDIM、DPM-Solver、EDM、consistency model 时，就能明白它们大多是在改进“怎么走这条路”。卡住时回到第 10 节、第 11 节和第 12 节。</p>
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
    "VAE",
    "Diffusion Model",
    "Generative Model",
    "Machine Learning",
    "Probability",
    "Maximum Likelihood",
    "KL Divergence",
    "Variational Inference",
    "ELBO",
    "Latent Variable Model",
    "DDPM",
    "Score Matching",
    "Flow Matching",
    "SDE",
    "ODE"
  ],
  categories: [{ name: "学习笔记", count: 3 }],
};
